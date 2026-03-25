import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Cargar variables
load_dotenv()

# 2. Configurar Groq (El cerebro de todos los empleados)
mi_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7 
)

# 3. La herramienta de búsqueda
@tool("BusquedaInternet")
def busqueda_internet(query: str):
    """Útil para buscar tendencias y noticias actuales en internet."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# ==========================================
#  EMPLEADO 1: El Estratega (Investigador)
# ==========================================
investigador = Agent(
    role='Estratega de Contenido Viral',
    goal='Encontrar las tendencias más fuertes de hoy sobre {tema}',
    backstory='Eres un experto en algoritmos de redes sociales que sabe qué está funcionando hoy.',
    tools=[busqueda_internet],
    llm=mi_llm,
    verbose=True,
    allow_delegation=False
)

# ==========================================
#  EMPLEADO 2: El Guionista (Redactor)
# ==========================================
guionista = Agent(
    role='Guionista Estrella de TikTok',
    goal='Escribir un guion dinámico y atractivo de 60 segundos basado en la investigación del Estratega.',
    backstory='Eres un redactor creativo famoso por hacer videos virales. Sabes exactamente dónde poner pausas, qué texto mostrar en pantalla y cómo mantener a la gente viendo hasta el último segundo.',
    llm=mi_llm,
    verbose=True,
    allow_delegation=False
)

# ==========================================
#  EMPLEADO 3: El Marketer (Distribución)
# ==========================================
marketer = Agent(
    role='Especialista en Growth Marketing',
    goal='Crear el paquete de publicación perfecto (Caption, CTA, y consejos de publicación) para que el video explote en vistas.',
    backstory='Eres un genio del marketing digital. Sabes cómo escribir descripciones que obligan a la gente a comentar y guardar el video. Entiendes los mejores horarios y tácticas de engagement en TikTok.',
    llm=mi_llm,
    verbose=True,
    allow_delegation=False
)

# ==========================================
# 📋 TAREAS
# ==========================================
tarea_investigacion = Task(
    description='Busca 3 temas virales y 5 hashtags clave sobre {tema}.',
    expected_output='Un reporte con Temas, Hashtags y una idea de hook.',
    agent=investigador
)

tarea_guion = Task(
    description='Lee el reporte del Estratega y escribe un guion de TikTok sobre {tema}. Usa el hook sugerido y desarrolla uno de los temas virales.',
    expected_output='Un guion de TikTok listo para grabar, con indicaciones de [Texto en Pantalla], [Audio/Efectos] y el texto que dice el presentador.',
    agent=guionista
)

tarea_marketing = Task(
    description='Revisa el guion final y los hashtags de la investigación. Escribe una descripción (caption) persuasiva para el video, incluye un fuerte "Llamado a la Acción" (CTA) para generar comentarios, añade los hashtags, y sugiere 2 estrategias rápidas para responder los primeros comentarios.',
    expected_output='Un paquete de publicación que incluya: Caption, Llamado a la acción, Hashtags, y consejos rápidos de engagement.',
    agent=marketer
)

# ==========================================
# 🏢 LA EMPRESA (Crew)
# ==========================================
crew = Crew(
    agents=[investigador, guionista, marketer], # <--- Los 3 empleados
    tasks=[tarea_investigacion, tarea_guion, tarea_marketing], # <--- Las 3 tareas en orden
    process=Process.sequential 
)

if __name__ == "__main__":
    print("--- El departamento de marketing ha entrado a la sala ---")
    resultado = crew.kickoff(inputs={'tema': 'Inteligencia Artificial para empresas'})
    
    print("\n\n##############################")
    print("## PAQUETE FINAL LISTO #######")
    print("##############################\n")
    print(resultado)