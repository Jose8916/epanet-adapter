import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from docx import Document
from fpdf import FPDF
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import multiprocessing as mp
from epanettools import epanet2 as et
from queue import Empty
from scipy.optimize import differential_evolution


def metodo_caracteristicas_simple(rho, c, L, D, v_inicial=1.0, t_cierre_valvula=1.0, escenario="sin_bomba", flujo_bomba=None, presion_bomba=None, t_total=5.0, dt=0.01, H_up=30.0, H_down=25.0):

    A = np.pi*(D**2)/4.0

    Q0 = v_inicial * A

    if escenario == "con_bomba" and flujo_bomba and presion_bomba:
        Q_bomba = flujo_bomba / 3600.0
        Q0 = Q_bomba 

    nsteps = int(t_total/dt) + 1
    t_array = np.linspace(0, t_total, nsteps)

    Hup = np.zeros(nsteps)
    Hdown = np.zeros(nsteps)
    Qline = np.zeros(nsteps)

    Hup[0] = H_up
    Hdown[0] = H_down
    Qline[0] = Q0

    g = 9.81
    f = 0.02
    Beta = f * L / (2*g*D)
    alpha = c/(g*A)  

    for n in range(1, nsteps):
        t_now = t_array[n]

        Hup[n] = H_up

        if t_now < t_cierre_valvula:
            valve_factor = 1.0 - (t_now / t_cierre_valvula)
        else:
            valve_factor = 0.0  

        Q_old = Qline[n-1]
        Hd_old = Hdown[n-1]

        c_plus = Hup[n] - Beta*Q_old*abs(Q_old) + alpha*Q_old

        c_minus = Hd_old + Beta*Q_old*abs(Q_old) - alpha*Q_old

        Qc = (c_plus - c_minus)/(2*Beta + 1e-12)

        Q_new = valve_factor * Qc

        H_d = c_minus + alpha*Q_new - Beta*Q_new*abs(Q_new)

        Qline[n] = Q_new
        Hdown[n] = H_d

    return t_array, Hup, Hdown, Qline


class BombaCentrifuga:
    def __init__(self, H0, Q0, potencia, inercia=100.0):
        self.H0 = H0  
        self.Q0 = Q0 
        self.potencia = potencia  
        self.inercia = inercia  
        self.vel_angular = 0.0  
        self.estado = False 
        
    def arrancar(self, dt):
        if not self.estado:
            torque = (self.potencia * 1000) / (self.vel_angular + 1e-5)
            self.vel_angular += (torque / self.inercia) * dt
            if self.vel_angular >= 1500 * (2*np.pi/60):  
                self.estado = True
                
    def parar(self, dt):
        if self.estado:
            torque_frenado = -self.vel_angular * 0.1 
            self.vel_angular += (torque_frenado / self.inercia) * dt
            if self.vel_angular <= 0:
                self.estado = False
                self.vel_angular = 0.0
                
    def curva_caracteristica(self, Q):
        return self.H0 - 0.2*self.H0*(Q/self.Q0)**2  


class ValvulaNoRetorno:
    def __init__(self, tiempo_cierre=1.0, umbral_flujo=0.01):
        self.tiempo_cierre = tiempo_cierre 
        self.umbral = umbral_flujo 
        self.abierta = True
        
    def actualizar_estado(self, flujo, dt):
        if self.abierta and flujo < -self.umbral: 
            self.tiempo_actual = 0.0
            self.abierta = False
        elif not self.abierta:
            self.tiempo_actual += dt
            if self.tiempo_actual >= self.tiempo_cierre:
                self.abierta = True
                
    def factor_apertura(self):
        return 1.0 if self.abierta else 0.0


def simular_red_epanet(archivo_inp, duracion=3600):
    try:

        ret = et.ENopen(archivo_inp, "reporte.txt", "")
        
        et.ENopenH()
        et.ENinitH(0) 
        
        tiempos = []
        presiones = {}
        flujos = {}
        
        while True:
            errcode, t = et.ENrunH() 
            tiempos.append(t)
            
            errcode, n_nodes = et.ENgetcount(et.EN_NODECOUNT)
            for idx in range(1, n_nodes + 1):
                errcode, presion = et.ENgetnodevalue(idx, et.EN_PRESSURE)
                presiones.setdefault(idx, []).append(presion)
            
            errcode, n_links = et.ENgetcount(et.EN_LINKCOUNT)
            for idx in range(1, n_links + 1):
                errcode, flujo = et.ENgetlinkvalue(idx, et.EN_FLOW)
                flujos.setdefault(idx, []).append(flujo)
            
            if t >= duracion:
                break
                
            et.ENnextH()
        
        et.ENcloseH()
        et.ENclose()
        
        return {
            'tiempos': np.array(tiempos),
            'presiones': presiones,
            'flujos': flujos
        }
        
    except Exception as e:
        et.ENclose()
        raise e

        
class ProblemaHidraulico(Problem):
    def __init__(self, red_epanet):
        super().__init__(n_var=3, n_obj=2, n_constr=1,
                         xl=np.array([0.1, 0.5, 1.0]),
                         xu=np.array([2.0, 5.0, 10.0]))
        self.red = red_epanet
        
    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2))
        G = np.zeros((X.shape[0], 1))
        
        for i in range(X.shape[0]):
            diam, v_ini, t_cierre = X[i]
            
            t, Hup, Hdown = metodo_caracteristicas_simple(
                L=self.red['L'], D=diam, v_inicial=v_ini, 
                t_cierre_valvula=t_cierre
            )
            
            costo = diam * self.red['L'] * 1500
            sobrepresion = np.max(Hdown) - self.red['H_down']
            
            F[i, 0] = costo
            F[i, 1] = sobrepresion
            
            G[i, 0] = 20 - np.min(Hdown)
            
        out["F"] = F
        out["G"] = G

def worker_simulacion(archivo, queue):
    try:
        data = leer_epanet_principal(archivo)
        if 'error' in data:
            queue.put(('error', data['error']))
            return
        
        queue.put(('parametros', data))
        
        resultados = simular_red_epanet(archivo)
        queue.put(('resultados', resultados))
        
    except Exception as e:
        queue.put(('error', str(e)))

def leer_epanet_principal(filename):
    params = {
        'rho': 1000.0,
        'c': 1200.0,
        'L': 100.0,
        'D': 0.5,
        'H_up': 30.0,
        'H_down': 25.0
    }
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith(";"):
                if match := re.search(r'rho\s*=\s*([\d\.]+)', line, re.IGNORECASE):
                    params['rho'] = float(match.group(1))
                if match := re.search(r'c\s*=\s*([\d\.]+)', line, re.IGNORECASE):
                    params['c'] = float(match.group(1))

    
    try:
        err = et.ENopen(filename, "report.txt", "")
        errcode, n_links = et.ENgetcount(et.EN_LINKCOUNT)
        selected_link_idx = None

        for idx in range(1, n_links + 1):
            errcode, link_type = et.ENgetlinktype(idx)
            if errcode == 0:
                errcode, length = et.ENgetlinkvalue(idx, et.EN_LENGTH)
                errcode, diameter = et.ENgetlinkvalue(idx, et.EN_DIAMETER)
                if errcode == 0:
                    params['L'] = length
                    params['D'] = diameter
                    selected_link_idx = idx
                    break

        if selected_link_idx is not None:
            errcode, node1, node2 = et.ENgetlinknodes(selected_link_idx)
            errcode, elev1 = et.ENgetnodevalue(node1, et.EN_ELEVATION)
            errcode, elev2 = et.ENgetnodevalue(node2, et.EN_ELEVATION)
            params['H_up'] = elev1 + 10.0
            params['H_down'] = elev2 + 5.0

        et.ENclose()
        return params
        
    except Exception as e:
        et.ENclose()
        return {'error': f"Error EPANET: {str(e)}"}
    

def metodo_caracteristicas_avanzado(rho, c, L, D,
                                    v_inicial=1.0,
                                    t_cierre_valvula=1.0,
                                    t_pump_start=1.0, t_pump_stop=2.0,
                                    scenario="sin_bomba",
                                    flujo_bomba=None,
                                    presion_bomba=None,
                                    t_total=5.0,
                                    dt=0.01,
                                    H_up=30.0,
                                    H_down=25.0):
    A = np.pi*(D**2)/4.0
    nsteps = int(t_total/dt) + 1
    t_array = np.linspace(0, t_total, nsteps)
    
    Hup = np.zeros(nsteps)
    Hdown = np.zeros(nsteps)
    Qline = np.zeros(nsteps)
    
    Hup[0] = H_up
    Hdown[0] = H_down
    Qline[0] = v_inicial * A

    g = 9.81
    f = 0.02
    Beta = f * L / (2*g*D)
    alpha = c/(g*A)

    for n in range(1, nsteps):
        t_now = t_array[n]
        Q_old = Qline[n-1]
        
        if scenario in ["sin_bomba", "cierre_valvula"]:
            if scenario == "sin_bomba":
                valve_factor = 1.0 - (t_now/t_cierre_valvula) if t_now < t_cierre_valvula else 0.0
            else:
                valve_factor = 1.0 if t_now < t_cierre_valvula else 0.0
            Hup[n] = H_up 
            c_plus = Hup[n] - Beta * Q_old * abs(Q_old) + alpha * Q_old
            c_minus = Hdown[n-1] + Beta * Q_old * abs(Q_old) - alpha * Q_old
            Qc = (c_plus - c_minus) / (2*Beta + 1e-12)
            Q_new = valve_factor * Qc
        elif scenario == "con_bomba":
            if t_now < t_pump_start:
                pump_factor = 0.0
            elif t_now < t_pump_stop:
                pump_factor = (t_now - t_pump_start) / (t_pump_stop - t_pump_start)
            else:
                pump_factor = 1.0
            pump_head = pump_factor * (presion_bomba/(rho*g)) if presion_bomba is not None else 0.0
            Hup[n] = H_up + pump_head
            c_plus = Hup[n] - Beta * Q_old * abs(Q_old) + alpha * Q_old
            c_minus = Hdown[n-1] + Beta * Q_old * abs(Q_old) - alpha * Q_old
            Q_new = (c_plus - c_minus) / (2*Beta + 1e-12)
        else:
            Hup[n] = H_up
            c_plus = Hup[n] - Beta * Q_old * abs(Q_old) + alpha * Q_old
            c_minus = Hdown[n-1] + Beta * Q_old * abs(Q_old) - alpha * Q_old
            Q_new = (c_plus - c_minus) / (2*Beta + 1e-12)

        Qline[n] = Q_new
        Hdown[n] = c_minus + alpha * Q_new - Beta * Q_new * abs(Q_new)

    return t_array, Hup, Hdown

class AppGolpeAriete(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EPANET-PYTHON con MOC - Avanzado")
        self.geometry("1380x900") 
        self.config(bg="#f0f8ff")
        self.resizable(False, False)
        self.progress_window = None  
        self.simulation_thread = None  
        self.simulation_process = None
        self.queue = mp.Queue()

        self.parametros = {
            'rho': {"label": "Densidad (kg/m³):", "entry": None},
            'c': {"label": "Velocidad de onda (m/s):", "entry": None},
            'L': {"label": "Longitud (m):", "entry": None},
            'D': {"label": "Diámetro (m):", "entry": None},
            'v_inicial': {"label": "Vel. inicial (m/s):", "entry": None},
            'tiempo_cierre_valvula': {"label": "Tiempo cierre válvula (s):", "entry": None},
            't_total': {"label": "Tiempo total sim (s):", "entry": None},
            'dt': {"label": "Paso de tiempo dt (s):", "entry": None}
        }

        self.flujo_bomba = None
        self.presion_bomba = None

        self.scenario = tk.StringVar(value="sin_bomba")

        self.crear_interfaz()
        self.graph_counter = 0

        self.H_up = 30.0
        self.H_down = 25.0

        self.bombas = []
        self.valvulas = []
        self.datos_epanet = None

    def cargar_archivo(self):
        archivo = filedialog.askopenfilename(
            title="Selecciona un archivo EPANET (.inp)",
            filetypes=[("Archivos EPANET", "*.inp"), ("Todos los archivos", "*.*")]
        )
        if not archivo:
            return

        self.mostrar_progreso()
        self.simulation_process = mp.Process(
            target=worker_simulacion,
            args=(archivo, self.queue),
            daemon=True
        )
        self.simulation_process.start()
        self.verificar_estado_proceso()

    def ejecutar_simulacion_epanet(self, archivo, queue):
        try:
            data = leer_epanet_principal(archivo)
            queue.put(('parametros', data))
            
            resultados = simular_red_epanet(archivo)
            queue.put(('resultados', resultados))
            
        except Exception as e:
            queue.put(('error', str(e)))

    def verificar_estado_proceso(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == 'parametros':
                    self.actualizar_parametros(msg[1])
                elif msg[0] == 'resultados':
                    self.datos_epanet = msg[1]
                    messagebox.showinfo("Éxito", "Simulación completada exitosamente")
                elif msg[0] == 'error':
                    messagebox.showerror("Error", msg[1])
        except Empty:
            if self.simulation_process.is_alive():
                self.after(100, self.verificar_estado_proceso)
            else:
                self.ocultar_progreso()


    def actualizar_parametros(self, data):
        if 'error' in data:
            messagebox.showerror("Error", data['error'])
            return
            
        for key in ['rho', 'c', 'L', 'D']:
            self.parametros[key]["entry"].delete(0, tk.END)
            self.parametros[key]["entry"].insert(0, str(data[key]))
        
        self.H_up = data['H_up']
        self.H_down = data['H_down']

    def mostrar_progreso(self):
        self.progress_window = tk.Toplevel(self)
        self.progress_window.title("Procesando...")
        
        frame = tk.Frame(self.progress_window, padx=20, pady=20)
        frame.pack()
        
        tk.Label(frame, text="Simulando red EPANET...", font=("Arial", 12)).pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            frame, 
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start(15)
        
        self.progress_window.geometry("+%d+%d" % (
            self.winfo_x() + self.winfo_width()/2 - 150,
            self.winfo_y() + self.winfo_height()/2 - 50
        ))
        self.progress_window.grab_set()

    def ocultar_progreso(self):
        if self.progress_window:
            self.progress_bar.stop()
            self.progress_window.destroy()
            self.progress_window = None

    def verificar_estado_hilo(self):
        if self.simulation_thread.is_alive():
            self.after(100, self.verificar_estado_hilo)
        else:
            self.after(0, self.fin_carga_archivo)

    def fin_carga_archivo(self):
        self.ocultar_progreso()
        messagebox.showinfo("Archivo cargado", "Simulación EPANET completada exitosamente")

    def validar_con_epanet(self):
        if not self.datos_epanet:
            messagebox.showerror("Error", "Primero carga un archivo EPANET.")
            return

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        params = {key: float(self.parametros[key]["entry"].get()) for key in ['rho', 'c', 'L', 'D']}
        t_moc, Hup_moc, Hdown_moc, Qline = metodo_caracteristicas_simple(**params)
        
        nodo_objetivo = 1 
        ax1.plot(self.datos_epanet['tiempos'], self.datos_epanet['presiones'][nodo_objetivo], 
                label='EPANET', linestyle='--')
        ax1.plot(t_moc, Hdown_moc, label='MOC')
        ax1.set_title("Comparación de Presiones")
        
        tuberia_objetivo = 1 
        ax2.plot(self.datos_epanet['tiempos'], self.datos_epanet['flujos'][tuberia_objetivo],
                label='EPANET', linestyle='--')
        ax2.plot(t_moc, [q * 1000 for q in Qline], label='MOC') 
        ax2.set_title("Comparación de Flujos")
        
        plt.legend()
        self.mostrar_grafico(fig)

    def optimizacion_avanzada(self):
        try:
            datos_red = {
                'L': float(self.parametros["L"]["entry"].get()),
                'H_down': self.H_down
            }
            
            problema = ProblemaHidraulico(datos_red)
            algoritmo = NSGA2(pop_size=50)
            res = minimize(problema, algoritmo, ('n_gen', 100))
            
            soluciones = res.X
            objetivos = res.F
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(objetivos[:, 0], objetivos[:, 1], c='red')
            ax.set_xlabel("Costo")
            ax.set_ylabel("Sobrepresión")
            self.mostrar_grafico(fig)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en optimización: {str(e)}")

    def cargar_caso_peruano(self):
        datos = {
            'rho': 1000,
            'c': 1200,
            'L': 2500,  
            'D': 0.6,   
            'H_up': 1500, 
            'H_down': 1450
        }
        
        for key in ['rho', 'c', 'L', 'D']:
            self.parametros[key]["entry"].delete(0, tk.END)
            self.parametros[key]["entry"].insert(0, str(datos[key]))
        
        self.H_up = datos['H_up']
        self.H_down = datos['H_down']
        messagebox.showinfo("Caso Peruano", "Parámetros de la Sierra cargados.")

    def mostrar_grafico(self, fig):
        ventana_grafico = tk.Toplevel(self)
        ventana_grafico.title("Resultados Avanzados")
        canvas = FigureCanvasTkAgg(fig, master=ventana_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def crear_interfaz(self):
        frame = tk.Frame(self, bg="#ffffff", padx=30, pady=30, relief="solid", bd=2)
        frame.place(relx=0.05, rely=0.5, anchor="w")

        titulo = tk.Label(frame, text="Parámetros", font=("Arial", 20, "bold"), bg="#ffffff", fg="#00796b")
        titulo.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        row_index = 1
        for key, info in self.parametros.items():
            label = tk.Label(frame, text=info["label"], bg="#ffffff", font=("Arial", 10, "italic"), anchor="w")
            label.grid(row=row_index, column=0, padx=10, pady=5, sticky="w")

            entry = tk.Entry(frame, font=("Arial", 10), width=20, relief="solid", bd=1,
                             highlightthickness=2, highlightbackground="#00796b")
            entry.grid(row=row_index, column=1, padx=10, pady=5)
            self.parametros[key]["entry"] = entry
            row_index += 1

        escenario_label = tk.Label(frame, text="Escenario:", bg="#ffffff", font=("Arial", 10, "italic"))
        escenario_label.grid(row=row_index, column=0, padx=10, pady=5, sticky="w")
        rb1 = tk.Radiobutton(frame, text="Base (Sin Bomba)", variable=self.scenario, value="sin_bomba", bg="#ffffff")
        rb2 = tk.Radiobutton(frame, text="Con Bomba (Arranque/Parada)", variable=self.scenario, value="con_bomba", bg="#ffffff")
        rb3 = tk.Radiobutton(frame, text="Cierre Repentino Válvula", variable=self.scenario, value="cierre_valvula", bg="#ffffff")
        rb1.grid(row=row_index, column=1, padx=10, pady=2, sticky="w")
        rb2.grid(row=row_index+1, column=1, padx=10, pady=2, sticky="w")
        rb3.grid(row=row_index+2, column=1, padx=10, pady=2, sticky="w")
        row_index += 3

        button_cargar = tk.Button(
            frame, text="Cargar archivo EPANET", command=self.cargar_archivo,
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat", padx=15, pady=10
        )
        button_cargar.grid(row=row_index, column=0, columnspan=2, pady=(20, 10))
        row_index += 1

        button_calcular = tk.Button(
            frame, text="Calcular y Graficar (MOC)", command=self.calcular,
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat", padx=15, pady=10
        )
        button_calcular.grid(row=row_index, column=0, columnspan=2, pady=10)
        row_index += 1

        button_analizar = tk.Button(
            self, text="Realizar Análisis", command=self.realizar_analisis,
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat", padx=15, pady=10
        )
        button_analizar.place(relx=0.1, rely=0.1, anchor="center")

        button_insertar_bomba = tk.Button(
            self, text="Insertar Bomba", command=self.insertar_bomba,
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat", padx=15, pady=10
        )
        button_insertar_bomba.place(relx=0.25, rely=0.1, anchor="center")

        button_opt = tk.Button(
            self, text="Optimizar Sistema", command=self.optimizar_sistema,
            font=("Arial", 12), bg="#FF5722", fg="white", relief="flat", padx=15, pady=10
        )
        button_opt.place(relx=0.4, rely=0.1, anchor="center")

        button_validar = tk.Button(
            self, text="Validar vs EPANET", command=self.validar_con_epanet,
            font=("Arial", 12), bg="#4CAF50", fg="white", relief="flat", padx=15, pady=10
        )
        button_validar.place(relx=0.55, rely=0.1, anchor="center")

        button_opt_avanzada = tk.Button(
            self, text="Optimización Avanzada", command=self.optimizacion_avanzada,
            font=("Arial", 12), bg="#FF5722", fg="white", relief="flat", padx=15, pady=10
        )
        button_opt_avanzada.place(relx=0.7, rely=0.1, anchor="center")

        button_caso_peru = tk.Button(
            self, text="Cargar Caso Peruano", command=self.cargar_caso_peruano,
            font=("Arial", 12), bg="#2196F3", fg="white", relief="flat", padx=15, pady=10
        )
        button_caso_peru.place(relx=0.85, rely=0.1, anchor="center")

        self.resultados_label = tk.Label(
            self, text="RESULTADOS", font=("Arial", 14, "bold"),
            bg="#ffffff", fg="#00796b"
        )
        self.resultados_label.place(relx=0.75, rely=0.1, anchor="e")

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.place(relx=0.7, rely=0.5, anchor="center")

        self.canvas = tk.Canvas(self.canvas_frame, bg="white", width=800, height=600,
                                scrollregion=(5200, 1200, 5200, 2000))
        self.scrollbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.canvas.pack()

        self.graph_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.graph_frame, anchor="nw")

    def insertar_bomba(self):
        ventana_bomba = tk.Toplevel(self)
        ventana_bomba.title("Parámetros de la Bomba")
        ventana_bomba.geometry("300x200")
        ventana_bomba.config(bg="#f0f8ff")
        
        parametros_bomba = {
            'flujo': {"label": "Flujo (m³/h):", "entry": None},
            'presion': {"label": "Presión (bar):", "entry": None},
            'potencia': {"label": "Potencia (kW):", "entry": None}
        }

        row_index = 1
        for key, info in parametros_bomba.items():
            label = tk.Label(ventana_bomba, text=info["label"], bg="#ffffff",
                             font=("Arial", 10), anchor="w")
            label.grid(row=row_index, column=0, padx=10, pady=5, sticky="w")
            entry = tk.Entry(ventana_bomba, font=("Arial", 10), width=20, relief="solid",
                             bd=1, highlightthickness=2, highlightbackground="#00796b")
            entry.grid(row=row_index, column=1, padx=10, pady=5)
            parametros_bomba[key]["entry"] = entry
            row_index += 1

        button_guardar = tk.Button(
            ventana_bomba, text="Guardar Parámetros",
            command=lambda: self.guardar_parametros_bomba(parametros_bomba),
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat", padx=15, pady=10
        )
        button_guardar.grid(row=row_index, column=0, columnspan=2, pady=10)

    def guardar_parametros_bomba(self, parametros_bomba):
        try:
            flujo = float(parametros_bomba["flujo"]["entry"].get())
            presion = float(parametros_bomba["presion"]["entry"].get())
            potencia = float(parametros_bomba["potencia"]["entry"].get())
            messagebox.showinfo(
                "Parámetros Guardados",
                f"Flujo: {flujo} m³/h\nPresión: {presion} bar\nPotencia: {potencia} kW"
            )
            self.flujo_bomba = flujo
            self.presion_bomba = presion
        except ValueError:
            messagebox.showerror(
                "Error",
                "Por favor, ingresa valores numéricos válidos para la bomba."
            )


    def calcular(self):
        try:
            rho = float(self.parametros["rho"]["entry"].get())
            c = float(self.parametros["c"]["entry"].get())
            L = float(self.parametros["L"]["entry"].get())
            D = float(self.parametros["D"]["entry"].get())
            v_inicial = float(self.parametros["v_inicial"]["entry"].get())
            t_cierre = float(self.parametros["tiempo_cierre_valvula"]["entry"].get())
            t_total = float(self.parametros["t_total"]["entry"].get()) if self.parametros["t_total"]["entry"].get() else 5.0
            dt = float(self.parametros["dt"]["entry"].get()) if self.parametros["dt"]["entry"].get() else 0.01

            escenario = self.scenario.get()
            t_pump_start = 1.0
            t_pump_stop = 2.0

            t1, hup1, hdown1 = metodo_caracteristicas_avanzado(
                rho, c, L, D,
                v_inicial=v_inicial,
                t_cierre_valvula=t_cierre,
                t_pump_start=t_pump_start, t_pump_stop=t_pump_stop,
                scenario=escenario,
                flujo_bomba=self.flujo_bomba,
                presion_bomba=self.presion_bomba,
                t_total=t_total,
                dt=dt,
                H_up=self.H_up,
                H_down=self.H_down
            )

            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(t1, hup1, label="Upstream")
            ax1.plot(t1, hdown1, label="Downstream")
            ax1.set_title(f"Escenario: {escenario}")
            ax1.set_xlabel("Tiempo (s)")
            ax1.set_ylabel("Altura (m)")
            ax1.legend()
            ax1.grid(True)
            canvas1 = FigureCanvasTkAgg(fig1, self.graph_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(pady=10, padx=10)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            messagebox.showinfo("Resultado", 
                                "Simulación completada con éxito.\nRevisa los gráficos para ver los resultados.")
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al calcular:\n{e}")

    def realizar_analisis(self):
        try:
            rho = float(self.parametros["rho"]["entry"].get())
            c = float(self.parametros["c"]["entry"].get())
            L = float(self.parametros["L"]["entry"].get())
            D = float(self.parametros["D"]["entry"].get())
            v_inicial = float(self.parametros["v_inicial"]["entry"].get())
            t_cierre = float(self.parametros["tiempo_cierre_valvula"]["entry"].get())
            t_array, hup, hdown = metodo_caracteristicas_avanzado(
                rho, c, L, D,
                v_inicial=v_inicial,
                t_cierre_valvula=t_cierre,
                scenario=self.scenario.get(),
                t_total=5.0,
                dt=0.01,
                H_up=self.H_up,
                H_down=self.H_down
            )
            g = 9.81
            p_up = rho * g * hup  
            p_down = rho * g * hdown
            max_p = max(np.max(p_up), np.max(p_down))
            base_p = rho * g * self.H_down
            delta_p = max_p - base_p
            analisis = self.generar_analisis(delta_p, rho, c, L, D, v_inicial)
            self.mostrar_analisis(analisis)
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def generar_analisis(self, delta_P, rho, c, L, D, v_inicial):
        analisis = "Análisis del Golpe de Ariete (MOC Avanzado)\n"
        analisis += "--------------------------------\n\n"
        analisis += f"Densidad (rho): {rho} kg/m³\n"
        analisis += f"Velocidad de onda (c): {c} m/s\n"
        analisis += f"Longitud (L): {L} m\n"
        analisis += f"Diámetro (D): {D} m\n"
        analisis += f"Velocidad inicial (v_inicial): {v_inicial} m/s\n\n"
        analisis += f"Sobrepresión máxima estimada (ΔP): {delta_P:.2f} Pa\n\n"

        if delta_P > 1e6:
            analisis += "El golpe de ariete es extremadamente alto. Riesgo de daños.\n"
            analisis += "Recomendaciones:\n"
            analisis += " - Amortiguadores de presión\n"
            analisis += " - Válvulas de cierre lento\n"
            analisis += " - Diseño reforzado de tuberías\n"
        elif delta_P > 1e5:
            analisis += "El golpe de ariete es significativo. Podrían darse daños moderados.\n"
            analisis += "Recomendaciones:\n"
            analisis += " - Válvulas de asiento\n"
            analisis += " - Amortiguadores de presión\n"
            analisis += " - Monitoreo frecuente\n"
        else:
            analisis += "El golpe de ariete es bajo. Bajo riesgo de daños inmediatos.\n"
            analisis += "Recomendaciones:\n"
            analisis += " - Monitoreo periódico\n"
            analisis += " - Válvulas de amortiguamiento\n"

        analisis += "\nConclusión:\n"
        analisis += "Este informe muestra la sobrepresión simulada utilizando un modelo avanzado que incluye efectos de arranque/parada de bomba y cierre repentino de válvula.\n"
        analisis += "Para mayor exactitud, sería conveniente modelar una red completa con múltiples elementos.\n"
        return analisis

    def mostrar_analisis(self, analisis):
        analisis_window = tk.Toplevel(self)
        analisis_window.title("Análisis del Golpe de Ariete")
        analisis_window.geometry("600x400")
        analisis_window.resizable(False, False)
        analisis_window.config(bg="#ffffff")
        texto_analisis = tk.Text(analisis_window, wrap=tk.WORD, width=70, height=15, font=("Arial", 10))
        texto_analisis.insert(tk.END, analisis)
        texto_analisis.config(state=tk.DISABLED)
        texto_analisis.pack(padx=20, pady=20)
        button_guardar_word = tk.Button(
            analisis_window, text="Guardar en Word",
            command=lambda: self.guardar_en_word(analisis),
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat"
        )
        button_guardar_word.pack(pady=5)
        button_guardar_pdf = tk.Button(
            analisis_window, text="Guardar en PDF",
            command=lambda: self.guardar_en_pdf(analisis),
            font=("Arial", 12), bg="#00796b", fg="white", relief="flat"
        )
        button_guardar_pdf.pack(pady=5)
        boton_cerrar = tk.Button(
            analisis_window, text="Cerrar",
            command=analisis_window.destroy,
            font=("Arial", 12), bg="#FF5722", fg="white", relief="flat"
        )
        boton_cerrar.pack(pady=10)

    def guardar_en_word(self, analisis):
        doc = Document()
        doc.add_heading('Análisis del Golpe de Ariete (MOC Avanzado)', 0)
        doc.add_paragraph(analisis)
        archivo_guardar = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Documentos Word", "*.docx")]
        )
        if archivo_guardar:
            doc.save(archivo_guardar)
            messagebox.showinfo("Guardado", f"El análisis se guardó en: {archivo_guardar}")

    def guardar_en_pdf(self, analisis):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Análisis del Golpe de Ariete (MOC Avanzado)", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, analisis)
        archivo_guardar = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Documentos PDF", "*.pdf")]
        )
        if archivo_guardar:
            pdf.output(archivo_guardar)
            messagebox.showinfo("Guardado", f"El análisis se guardó en: {archivo_guardar}")

    def optimizar_sistema(self):
        messagebox.showinfo(
            "Optimización",
            "Se buscará la combinación (v_inicial, t_cierre) que minimice la sobrepresión máxima."
        )
        bounds = [(0.0, 5.0), (0.1, 5.0)]
        try:
            rho = float(self.parametros["rho"]["entry"].get())
            c = float(self.parametros["c"]["entry"].get())
            L = float(self.parametros["L"]["entry"].get())
            D = float(self.parametros["D"]["entry"].get())
        except:
            messagebox.showerror(
                "Error",
                "Por favor ingresa parámetros válidos (rho, c, L, D) antes de optimizar."
            )
            return
        def objetivo(vars_):
            v_ini = vars_[0]
            t_cie = vars_[1]
            t, hup, hdown = metodo_caracteristicas_avanzado(
                rho, c, L, D,
                v_inicial=v_ini,
                t_cierre_valvula=t_cie,
                scenario=self.scenario.get(),
                H_up=self.H_up,
                H_down=self.H_down,
                t_total=5.0,
                dt=0.01
            )
            g = 9.81
            p_up = rho * g * hup
            p_down = rho * g * hdown
            max_p = max(np.max(p_up), np.max(p_down))
            return max_p
        result = differential_evolution(objetivo, bounds, maxiter=20, popsize=15, tol=1e-3)
        if result.success:
            v_opt_inicial, t_opt_cierre = result.x
            pres_min = result.fun
            msg = (f"Optimización finalizada con éxito.\n\n"
                   f"Vel. inicial óptima: {v_opt_inicial:.3f} m/s\n"
                   f"Tiempo de cierre óptimo: {t_opt_cierre:.3f} s\n"
                   f"Presión máxima final: {pres_min:.2f} Pa\n")
            messagebox.showinfo("Resultado de la Optimización", msg)
        else:
            messagebox.showwarning(
                "Optimización incompleta",
                "El optimizador no convergió a una solución satisfactoria."
            )

if __name__ == "__main__":
    app = AppGolpeAriete()
    app.mainloop()
