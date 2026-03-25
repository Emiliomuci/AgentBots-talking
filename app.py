import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Creador de Equipos IA", page_icon="🧠", layout="wide")
load_dotenv()

# --- ESTILOS LIMPIOS ---
st.markdown("""
    <style>
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .sub-title { text-align: center; color: #6b7280; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# --- MEMORIA DE LA PÁGINA ---
if "num_agentes" not in st.session_state:
    st.session_state.num_agentes = 2 

def agregar_agente():
    st.session_state.num_agentes += 1

def quitar_agente():
    if st.session_state.num_agentes > 1:
        st.session_state.num_agentes -= 1

# --- CABECERA ---
st.markdown("<h1 class='main-title'>Constructor de Equipos IA</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Arma tu agencia, define roles y controla el consumo de datos de cada bot.</p>", unsafe_allow_html=True)

# --- CONTROLES PARA AGREGAR/QUITAR BOTS ---
col_btn1, col_btn2, col_espacio = st.columns([1, 1, 4])
with col_btn1:
    st.button("➕ Agregar Agente", on_click=agregar_agente, use_container_width=True)
with col_btn2:
    st.button("➖ Quitar Agente", on_click=quitar_agente, use_container_width=True)

st.divider()

# --- FORMULARIO DINÁMICO DE AGENTES ---
agentes_config = [] 

st.subheader("🤖 Configura tu Equipo")
columnas_agentes = st.columns(st.session_state.num_agentes)

for i in range(st.session_state.num_agentes):
    with columnas_agentes[i]:
        with st.container(border=True):
            st.markdown(f"**Agente {i+1}**")
            rol = st.text_input(f"Rol del Agente", key=f"rol_{i}", placeholder="Ej. Analista de Datos")
            experiencia = st.text_area(f"Experiencia / Personalidad", key=f"exp_{i}", placeholder="Ej. Experto en leer mercados inmobiliarios...")
            tarea = st.text_area(f"¿Qué tarea específica hará?", key=f"tarea_{i}", placeholder="Ej. Busca tendencias actuales sobre {tema}")
            
            # 🔥 LA NUEVA MAGIA: El botón de Internet
            usar_internet = st.checkbox("🌐 Darle acceso a Internet", key=f"internet_{i}", help="Actívalo si necesita buscar noticias actuales. Mantenlo apagado para ahorrar tokens.")
            
            agentes_config.append({
                "rol": rol,
                "experiencia": experiencia,
                "tarea": tarea,
                "usar_internet": usar_internet # Guardamos tu decisión
            })

st.divider()

# --- MISIÓN PRINCIPAL ---
st.subheader("🎯 Misión Principal")
st.info("💡 Tip: En las tareas de tus agentes, asegúrate de escribir la palabra **{tema}** para que la IA sepa dónde insertar tu misión principal.")
tema_principal = st.text_area("¿De qué trata el proyecto general?", placeholder="Ej. Busco estrategias para comprar y vender casas...")

# --- EJECUCIÓN ---
if st.button("🚀 Iniciar Operación", type="primary", use_container_width=True):
    
    if not tema_principal.strip():
        st.error("Por favor, escribe la misión principal.")
    elif any(not a['rol'] or not a['tarea'] for a in agentes_config):
        st.error("Asegúrate de llenar el Rol y la Tarea de todos tus agentes.")
    else:
        with st.status("Ensamblando equipo y ejecutando misión...", expanded=True) as status:
            
            # 1. Configurar Cerebro Oficial (Corregido a LLM nativo)
            mi_llm = LLM(
                model="groq/llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3 
            )

            @tool("BusquedaInternet")
            def busqueda_internet(query: str):
                """Útil para buscar datos reales en internet."""
                search = DuckDuckGoSearchRun()
                return search.run(query)

            # 2. Crear las listas de CrewAI
            lista_agentes_crew = []
            lista_tareas_crew = []

            for config in agentes_config:
                
                # 🔥 EVALUAMOS TU DECISIÓN: ¿Le damos la herramienta de internet o se la quitamos?
                herramientas_del_agente = [busqueda_internet] if config['usar_internet'] else []
                
                nuevo_agente = Agent(
                    role=config['rol'],
                    goal=f"Tu objetivo principal es: {tema_principal}. REGLA CRÍTICA: NO inventes datos numéricos ni empresas que no se te hayan dado.",
                    backstory=f"{config['experiencia']}. Eres un consultor profesional. Limítate a planificar estrategias basadas en tu experiencia.",
                    tools=herramientas_del_agente, # <--- Aquí le entregamos (o no) el navegador
                    llm=mi_llm,
                    allow_delegation=False
                )
                lista_agentes_crew.append(nuevo_agente)

                nueva_tarea = Task(
                    description=f"{config['tarea']}. (Basado en el tema: {tema_principal}).",
                    expected_output="Documento estratégico y lógico. Prohibido incluir estadísticas inventadas.",
                    agent=nuevo_agente
                )
                lista_tareas_crew.append(nueva_tarea)

            # 3. Ensamblar la Empresa
            crew = Crew(
                agents=lista_agentes_crew,
                tasks=lista_tareas_crew,
                process=Process.sequential,
                max_rpm=1, # Freno anti-choques de Groq
                verbose=False,
                memory=True,
                embedder={
                    "provider": "huggingface",
                    "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
                }
            )

            st.write(f"Sincronizando a {st.session_state.num_agentes} agentes...")
            resultado = crew.kickoff(inputs={'tema': tema_principal})
            
            status.update(label="Operación exitosa", state="complete", expanded=False)
        
        # --- MOSTRAR RESULTADO ---
        st.markdown("### 📄 Resultado Final")
        with st.container(border=True):
            st.markdown(str(resultado))