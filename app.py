"""
Nombre: simulador_controlador.py (versión Web)
Autor: Oscar Franco
Versión: 1 (2024-09-17)
Descripción: Aplicación para simular el comportamiento de un sistema según su función de transferencia
            en lazo abierto o aplicando un controlador PID, ahora migrado a una aplicación web usando Streamlit.
"""

import numpy as np
import plotly.graph_objs as go
from scipy.integrate import solve_ivp
import streamlit as st

# Clase para la simulación del controlador PID
class SimuladorControlador:

    configuracion = {
            "variance": 5e-09,
            "tiempo_simulacion": 100,
            "ruidoSenalEncendido": True,
            "Ts": 0.1,
            "controlAutomaticoEncendido": True,
            "tminGrafica": 120,
            "Kp": 2.34,
            "taup": 132.78,
            "td": 9.10,
            "Kc": 1.0,
            "Ki": 0.0,
            "Kd": 0.0,
            "t0": 0,
            "y0": 50,
            "co0": 50,
            "u0": 0,
            "ysp_step": 52,
            "co_step": 55,
            "t_step": 10
        }
    
    if False:
        # Configuración sintonizada
        configuracion['Kc'] = 3.12
        configuracion['Ki'] = 0.040
        configuracion['Kd'] = 0.0

    def __init__(self):
        self.inicializar_parametros()
        self.inicializar_estado_simulacion()

    # Inicialización de parámetros de simulación desde la configuración
    def inicializar_parametros(self):
        config = self.configuracion
        self.variance = config['variance']
        self.tiempo_simulacion = config['tiempo_simulacion']
        self.Ts = config['Ts']
        self.controlAutomaticoEncendido = config['controlAutomaticoEncendido']
        self.tminGrafica = config['tminGrafica']
        self.Kp = config['Kp']
        self.taup = config['taup']
        self.td = config['td']
        self.Kc = config['Kc']
        self.Ki = config['Ki']
        self.Kd = config['Kd']

    # Inicializar variables de estado
    def inicializar_estado_simulacion(self):
        self.ruidoSenalEncendido = self.configuracion['ruidoSenalEncendido']

        self.t0 = self.configuracion['t0']
        self.y0 = self.configuracion['y0']
        self.co0 = self.configuracion['co0']
        self.u0 = self.configuracion['u0']
        self.ysp_step = self.configuracion['ysp_step']
        self.co_step = self.configuracion['co_step']
        self.t_step = self.configuracion['t_step']

        self.Ek2 = 0
        self.Ek1 = 0
        self.Ek = 0

        self.t = [0]
        self.y = [self.y0]
        self.co = [self.co0]
        self.ysp = [self.y0]
        self.nDatosGrafica = round(self.tminGrafica / self.Ts)

    # Definición del modelo FOPDT
    def fopdt(self, t, y, co):
        u = 0 if t < self.td + self.t_step else 1
        dydt = -(y - self.y0) / self.taup + self.Kp / self.taup * (u - self.u0) * (co - self.co0)
        return dydt

    # Simulación del lazo de control PID
    def simulacion_pid(self):
        try:
            # Cálculo de las constantes del controlador discretizado
            q0 = self.Kc + self.Ts * self.Ki / 2 + self.Kd / self.Ts
            q1 = self.Kc - self.Ts * self.Ki / 2 + 2 * self.Kd / self.Ts
            q2 = self.Kd / self.Ts

            # Inicializa las variables para controlar el bucle
            td_index_offset = int(self.td / self.Ts)  # Control retrasado
            mean_sqrt_variance = np.sqrt(self.variance * self.ruidoSenalEncendido)  # Precalculo de la varianza

            for i in range(self.tiempo_simulacion):

                self.t.append(self.t[-1] + self.Ts)

                ts = [self.t[-2], self.t[-1]]

                if self.t[-1]<self.t_step:
                    self.ysp.append(self.ysp[-1])
                else:
                    self.ysp.append(self.ysp_step)

                if i >= td_index_offset:
                    coAtrasado = self.co[-td_index_offset]  # Control retrasado
                else:
                    coAtrasado = self.co0

                sol = solve_ivp(self.fopdt, ts, [self.y[-1]], method='RK45', t_eval=[ts[-1]], args=(coAtrasado,))
                self.y.append(float(sol.y[0][-1]) * np.random.normal(1, mean_sqrt_variance))

                # Control PID
                self.Ek2, self.Ek1 = self.Ek1, self.Ek
                self.Ek = self.ysp[-1] - self.y[-1]
                deltaCO = q0 * self.Ek - q1 * self.Ek1 + q2 * self.Ek2
                self.co.append(max(0, min(100, self.co[-1] + deltaCO)))  # Limitar entre 0 y 100

            self.actualizar_grafica()

        except Exception as e:
            st.error(f"Error en la simulación: {e}")

    # Simulación del sistema en lazo abierto
    def simulacion_lazo_abierto(self):
        try:

            td_index_offset = int(self.td / self.Ts)  # Control retrasado
            mean_sqrt_variance = np.sqrt(self.variance * self.ruidoSenalEncendido)  # Precalculo de la varianza

            for i in range(self.tiempo_simulacion):

                self.t.append(self.t[-1] + self.Ts)

                ts = [self.t[-2], self.t[-1]]

                if self.t[-1]<self.t_step:
                    self.ysp.append(self.ysp[-1])
                    self.co.append(self.co[-1])
                else:
                    self.ysp.append(self.ysp_step)
                    self.co.append(self.co_step)

                if i >= td_index_offset:
                    coAtrasado = self.co[-td_index_offset]  # Control retrasado
                else:
                    coAtrasado = self.co0

                sol = solve_ivp(self.fopdt, ts, [self.y[-1]], method='RK45', t_eval=[ts[-1]], args=(coAtrasado,))
                self.y.append(float(sol.y[0][-1]) * np.random.normal(1, mean_sqrt_variance))

            self.actualizar_grafica()

        except Exception as e:
            st.error(f"Error en la simulación: {e}")

    def simulacion_sistema(self):
        if self.controlAutomaticoEncendido:
            self.simulacion_pid()
        else:
            self.simulacion_lazo_abierto()

    # Actualizar la gráfica de resultados
    def actualizar_grafica(self):
        fig = go.Figure()

        # Gráfico de la salida (y)
        fig.add_trace(go.Scatter(x=self.t, y=self.y, mode='lines', name='y', line=dict(color='blue')))
        if self.controlAutomaticoEncendido:
            fig.add_trace(go.Scatter(x=self.t, y=self.ysp, mode='lines', name='y_sp', line=dict(color='white')))

        # Gráfico del control (CO)
        fig.add_trace(go.Scatter(x=self.t, y=self.co, mode='lines', name='CO', yaxis='y2', line=dict(color='red')))

        # Ajustar el rango del eje X y el eje Y manualmente
        x_min = 0  # Valor mínimo para el eje x
        x_max = self.t[-1]  # Valor máximo para el eje x (último valor de tiempo)
        
        y_min = min(self.y + self.ysp) * 0.98  # Mínimo de las variables 'y' y 'ysp' con un margen
        y_max = max(self.y + self.ysp) * 1.02  # Máximo de las variables 'y' y 'ysp' con un margen
        
        co_min = round(min(self.co))*0.98  # Mínimo para el eje secundario 'CO'
        co_max = round(max(self.co))*1.02  # Máximo para el eje secundario 'CO' con un margen


        # Configuración de los ejes
        fig.update_layout(
            xaxis_title='Tiempo [s]',
            yaxis_title='Salida (y)',
            xaxis=dict(
            range=[x_min, x_max]  # Establece el rango del eje X
            ),
            yaxis=dict(
                title='Salida (y)',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                range=[y_min, y_max]  # Establece el rango del eje Y
            ),
            yaxis2=dict(
                title='CO [%]',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right',
                range=[co_min, co_max]  # Establece el rango del eje Y secundario
            ),
            legend=dict(x=0.8, y=0.05),
            template='plotly_white'
        )

        st.plotly_chart(fig)

# Crear una instancia del simulador
simulador = SimuladorControlador()

# Interfaz con Streamlit
st.title('Simulador de Sistemas de Control - Controlador PID')

# Parámetros del controlador PID
st.sidebar.header('Parámetros del Controlador PID')
Kc = st.sidebar.number_input('Kc', value=simulador.Kc, format="%.2f")
Ki = st.sidebar.number_input('Ki', value=simulador.Ki, format="%.3f")
Kd = st.sidebar.number_input('Kd', value=simulador.Kd, format="%.3f")
simulador.Kc = Kc
simulador.Ki = Ki
simulador.Kd = Kd

# Setpoint y velocidad
setpoint_step = st.sidebar.number_input('Set Point Step', value=simulador.ysp_step)
co_step = st.sidebar.number_input('CO Step', value=simulador.co_step)
tiempo_simulacion = st.sidebar.slider('Tiempo de simulación', min_value=10, max_value=500, value=simulador.tiempo_simulacion)*10
simulador.ysp_step = setpoint_step
simulador.co_step = co_step
simulador.tiempo_simulacion = tiempo_simulacion

# Control automático y ruido en señal
ruido = st.sidebar.checkbox('Simular señal ruidosa', value=simulador.ruidoSenalEncendido)
control_automatico = st.sidebar.checkbox('Control automático', value=simulador.controlAutomaticoEncendido)
simulador.ruidoSenalEncendido = ruido
simulador.controlAutomaticoEncendido = control_automatico

# Parámetros del sistema
password = st.sidebar.text_input('Unlock', type='password')
if password == 'unlockdinamica':
    st.sidebar.header('Parámetros del Sistema')
    Kp = st.sidebar.number_input('Kp', value=simulador.Kp, format="%.2f")
    taup = st.sidebar.number_input('Taup', value=simulador.taup, format="%.2f")
    td = st.sidebar.number_input('Td', value=simulador.td, format="%.2f")
    simulador.Kp = Kp
    simulador.taup = taup
    simulador.td = td

# Botón para iniciar simulación
if st.button('Iniciar Simulación'):

    simulador.simulacion_sistema()

# Mostrar la tabla de resultados
st.header('Resultados de la Simulación')
st.write('A continuación se muestra una tabla con los resultados de la simulación:')
st.dataframe({
    'Tiempo [s]': simulador.t,
    'Salida (y)': simulador.y,
    'Control (CO)': simulador.co,
    'Set Point': simulador.ysp
})
