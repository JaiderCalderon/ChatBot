from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

import dotenv
dotenv.load_dotenv()

# =========================
# 1. Cliente de OpenAI
# =========================
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# =========================
# 2. Modelo de embeddings
# =========================
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 3. Base de información
# =========================
def base_de_informacion():
    informacion_de_ciudades = [
        {
            "ciudad": "Cartagena",
            
            "descripcion": "una ciudad turística de playa con historia colonial y clima cálido",
            
            "lugares": """la Ciudad Amurallada, un espacio histórico lleno de calles coloniales, balcones coloridos y plazas que reflejan la historia de la ciudad; 
            el Castillo de San Felipe, una fortaleza emblemática que permite conocer la arquitectura militar y ofrece vistas panorámicas; 
            Playa Blanca, reconocida por su arena blanca y aguas tranquilas ideales para descansar; 
            las Islas del Rosario, perfectas para disfrutar de paisajes naturales, actividades acuáticas y contacto con la fauna marina; 
            Getsemaní, un barrio cultural lleno de arte urbano, música y ambiente local; 
            el Cerro de la Popa, desde donde se obtiene una vista completa de Cartagena; 
            la Torre del Reloj, uno de los principales puntos de acceso al centro histórico; 
            el Museo del Oro Zenú, que muestra la riqueza cultural indígena; 
            la Plaza Santo Domingo, ideal para disfrutar de la gastronomía y el ambiente turístico; 
            y las Bóvedas, un lugar histórico donde hoy se pueden encontrar artesanías y recuerdos""",
            
            "costo": "una de las ciudades más costosas de Colombia",
            
            "recomendacion": """se recomienda visitar la Ciudad Amurallada para quienes disfrutan la historia y la arquitectura colonial, ya que permite recorrer espacios llenos de cultura y tradición; 
            el Castillo de San Felipe es ideal para quienes desean conocer el pasado militar de la ciudad y disfrutar de vistas panorámicas; 
            Playa Blanca es perfecta para quienes buscan descanso, sol y mar en un ambiente tranquilo; 
            las Islas del Rosario son recomendadas para quienes quieren una experiencia más completa con actividades acuáticas y contacto con la naturaleza; 
            Getsemaní es ideal para quienes buscan arte, cultura y vida local; 
            el Cerro de la Popa es perfecto para obtener vistas panorámicas de la ciudad; 
            la Torre del Reloj es recomendada como punto de inicio para recorrer el centro histórico; 
            el Museo del Oro Zenú es ideal para quienes desean aprender sobre la cultura indígena; la Plaza Santo Domingo es 
            perfecta para disfrutar de la gastronomía y el ambiente turístico; y las Bóvedas son recomendadas para quienes quieren comprar recuerdos y artesanías locales"""
        },
        {
            "ciudad": "Medellín",
            
            "descripcion": "una ciudad moderna rodeada de montañas con clima agradable",
            
            "lugares": """la Comuna 13, reconocida por su arte urbano, sus escaleras eléctricas y su historia de transformación social; 
            el Pueblito Paisa, que ofrece una vista panorámica de la ciudad y una muestra de la arquitectura tradicional antioqueña; 
            el Jardín Botánico, ideal para disfrutar de la naturaleza y espacios tranquilos; el Parque Explora, un museo interactivo de ciencia y tecnología; 
            el Museo de Antioquia, famoso por albergar obras de Fernando Botero; el Parque Arví, una reserva natural para caminatas y ecoturismo; 
            el Metrocable, que permite observar la ciudad desde una perspectiva diferente; 
            el Estadio Atanasio Girardot, referente deportivo y cultural; 
            el Parque Lleras, conocido por su entretenimiento y vida nocturna; 
            y el Centro Comercial Santa Fe, uno de los más grandes y visitados de la ciudad""",
            
            "costo": "una ciudad de costo medio",
            
            "recomendacion": """se recomienda visitar la Comuna 13 para quienes desean conocer un lugar cargado de arte urbano, cultura e historia social; 
            el Pueblito Paisa es ideal para quienes quieren apreciar una vista panorámica de Medellín y vivir una experiencia tradicional; 
            el Jardín Botánico es perfecto para quienes buscan tranquilidad y contacto con la naturaleza; 
            el Parque Explora es recomendado para familias y personas interesadas en experiencias interactivas; 
            el Museo de Antioquia es ideal para los amantes del arte y la cultura; 
            el Parque Arví es perfecto para quienes disfrutan del ecoturismo y los paisajes naturales; 
            el Metrocable es una excelente opción para ver la ciudad desde otra perspectiva; 
            el Estadio Atanasio Girardot es recomendado para quienes disfrutan del deporte; 
            el Parque Lleras es ideal para quienes buscan entretenimiento y vida nocturna; y el Centro Comercial Santa Fe es una buena opción para compras y planes urbanos"""
        },
        {
            "ciudad": "Santa Marta",
            
            "descripcion": "una ciudad costera ideal para naturaleza y playa",
            
            "lugares": """el Parque Tayrona, reconocido por su biodiversidad, senderos ecológicos y playas paradisíacas; 
            Minca, un destino de montaña con cascadas, vegetación y clima fresco; 
            Playa Cristal, famosa por sus aguas transparentes y belleza natural; 
            Taganga, un pueblo pesquero con ambiente turístico y opciones de buceo; 
            la Quinta de San Pedro Alejandrino, un lugar histórico relacionado con Simón Bolívar; 
            el Rodadero, una de las playas más visitadas por turistas; 
            la Sierra Nevada de Santa Marta, ideal para el turismo ecológico y de aventura; 
            Ciudad Perdida, un sitio arqueológico rodeado de naturaleza; 
            el Acuario de Santa Marta, que permite conocer fauna marina; 
            y el centro histórico, que combina tradición, comercio y cultura local""",
            
            "costo": "una ciudad económica en comparación con otros destinos de playa",
            
            "recomendacion": """se recomienda visitar el Parque Tayrona para quienes buscan una experiencia completa de naturaleza, playa y caminatas ecológicas; 
            Minca es ideal para viajeros que disfrutan la montaña, el ecoturismo y el clima fresco; 
            Playa Cristal es perfecta para quienes desean disfrutar de aguas limpias y un ambiente tranquilo; 
            Taganga es recomendada para quienes buscan buceo y un ambiente relajado cerca del mar; 
            la Quinta de San Pedro Alejandrino es ideal para quienes desean aprender sobre historia; 
            el Rodadero es perfecto para planes familiares y turismo de playa; 
            la Sierra Nevada es recomendada para quienes buscan aventura y conexión con la naturaleza; 
            Ciudad Perdida es ideal para viajeros interesados en senderismo y experiencias arqueológicas; 
            el Acuario de Santa Marta es una buena opción para quienes quieren conocer la fauna marina; 
            y el centro histórico es perfecto para caminar, explorar la cultura local y disfrutar del ambiente de la ciudad"""
        },
        {
            "ciudad": "San Andrés",
            
            "descripcion": "una isla con playas de agua cristalina y mar de siete colores",
            
            "lugares": """Johnny Cay, una pequeña isla turística famosa por sus playas de arena blanca y ambiente caribeño; 
            el Acuario, una zona de mar poco profundo ideal para observar peces y disfrutar del agua; el Hoyo Soplador, un atractivo natural muy conocido por los visitantes; 
            West View, un lugar perfecto para nadar y saltar al mar; Spratt Bight, la playa principal de la isla con fácil acceso y ambiente turístico;
            la Cueva de Morgan, relacionada con historias de piratas y tradición local; 
            San Luis, una zona más tranquila y menos concurrida; 
            Rocky Cay, una pequeña isla cercana ideal para fotografías y caminatas; 
            la Piscinita, un lugar popular para hacer snorkel; 
            y el malecón, perfecto para recorrer la isla y disfrutar del paisaje""",
            
            "costo": "un destino costoso por transporte y hospedaje limitados en la isla",
            
            "recomendacion": """se recomienda visitar Johnny Cay para quienes desean una experiencia típica del Caribe con playa, arena blanca y ambiente relajado; 
            el Acuario es ideal para quienes disfrutan el contacto con el mar y la observación de peces; 
            el Hoyo Soplador es perfecto para quienes quieren conocer uno de los fenómenos naturales más llamativos de la isla; 
            West View es recomendado para quienes disfrutan nadar y vivir una experiencia más activa en el mar; 
            Spratt Bight es ideal para quienes buscan comodidad, acceso fácil y una playa representativa de San Andrés; 
            la Cueva de Morgan es recomendada para quienes quieren conocer historias locales y tradiciones de la isla; 
            San Luis es perfecto para quienes buscan tranquilidad; 
            Rocky Cay es una excelente opción para quienes desean tomar fotografías y disfrutar del paisaje; 
            la Piscinita es ideal para quienes quieren hacer snorkel; 
            y el malecón es perfecto para caminar, relajarse y disfrutar del ambiente isleño"""
        },
        {
            "ciudad": "Cali",
            
            "descripcion": "una ciudad cultural conocida por la salsa y su ambiente alegre y cálido todo el año",
            
            "lugares": """el Barrio San Antonio, reconocido por su arquitectura tradicional, ambiente cultural y oferta gastronómica; 
            el Cristo Rey, que ofrece una vista panorámica de la ciudad; 
            el Zoológico de Cali, uno de los más importantes del país; 
            el Bulevar del Río, ideal para caminar y disfrutar del ambiente urbano; 
            la Iglesia La Ermita, uno de los íconos arquitectónicos del centro; 
            el Parque del Perro, famoso por sus restaurantes y vida nocturna; 
            el Estadio Pascual Guerrero, referente deportivo de la ciudad; 
            el Museo La Tertulia, importante espacio cultural y artístico; 
            la Plaza de Cayzedo, un punto histórico y simbólico del centro de Cali; 
            y el río Pance, una zona natural muy visitada para descanso y recreación""",
            
            "costo": "una de las ciudades más económicas para viajar en Colombia",
            
            "recomendacion": """se recomienda visitar el Barrio San Antonio para quienes buscan una experiencia cultural, gastronómica y artística en un entorno tradicional; 
            el Cristo Rey es ideal para quienes desean observar Cali desde un punto panorámico y tomar buenas fotografías; 
            el Zoológico de Cali es perfecto para familias y amantes de los animales; 
            el Bulevar del Río es recomendado para quienes buscan caminar, relajarse y disfrutar del ambiente urbano; 
            la Iglesia La Ermita es ideal para quienes valoran la arquitectura y los lugares emblemáticos; 
            el Parque del Perro es perfecto para quienes disfrutan de la gastronomía y la vida nocturna; 
            el Estadio Pascual Guerrero es recomendado para quienes sienten interés por el deporte y el fútbol; 
            el Museo La Tertulia es ideal para amantes del arte contemporáneo; 
            la Plaza de Cayzedo es una excelente opción para conocer el centro histórico de la ciudad; 
            y el río Pance es perfecto para quienes buscan naturaleza, frescura y un plan de descanso al aire libre"""
        }
    ]
    return informacion_de_ciudades

# =========================
# 4. Generar chunks
# =========================
def generar_los_chunks():
    base_de_datos = base_de_informacion()
    todos_los_chunks = []

    for ciudad_info in base_de_datos:
        ciudad = ciudad_info["ciudad"]

        chunks = [
            f"{ciudad} es {ciudad_info['descripcion']}.",
            f"En {ciudad} se pueden visitar lugares como {ciudad_info['lugares']}.",
            f"En términos de costo, {ciudad} es {ciudad_info['costo']}.",
            f"Una recomendación en {ciudad} es: {ciudad_info['recomendacion']}."
        ]

        for chunk in chunks:
            todos_los_chunks.append({
                "ciudad": ciudad,
                "chunk": chunk
            })

    return todos_los_chunks

# =========================
# 5. Generar embedding
# =========================
def generar_embedding(chunks_y_pregunta):
    return modelo_embeddings.encode(chunks_y_pregunta)

# =========================
# 6. Construir base semántica
# =========================
def construir_base_semantica():
    chunks = generar_los_chunks()
    base_semantica = []

    for item in chunks:
        # Mostrar parte del chunk para seguimiento
        embedding = generar_embedding(item["chunk"])

        base_semantica.append({
            "ciudad": item["ciudad"],
            "chunk": item["chunk"],
            "embedding": embedding
        })

    return base_semantica

# =========================
# 7. Buscar chunks relevantes
# =========================
def buscar_chunks_relevantes(pregunta, base_semantica, top_k=5):
    embedding_pregunta = generar_embedding(pregunta)
    resultados = []

    for item in base_semantica:
        score = cosine_similarity(
            [embedding_pregunta],
            [item["embedding"]]
        )[0][0]

        resultados.append({
            "ciudad": item["ciudad"],
            "chunk": item["chunk"],
            "score": score
        })

    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados[:top_k]

# =========================
# 8. Construir contexto
# =========================
def construir_contexto(resultados):
    contexto = ""
    for item in resultados:
        contexto += f"- {item['chunk']}\n"
    return contexto

# =========================
# 9. Construir prompt
# =========================
def construir_prompt(contexto, pregunta):
    prompt = f"""
Eres un asistente de viajes.

Responde únicamente con la información del contexto.
Si la respuesta no está en el contexto, di:
"No encontré información suficiente para responder."

Contexto:
{contexto}

Pregunta del usuario:
{pregunta}

Respuesta:
"""
    return prompt

# =========================
# 10. Generar respuesta final
# =========================
def generar_respuesta(prompt):
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=120
    )
    return respuesta.choices[0].message.content

# =========================
# 11. Programa principal
# =========================
def main():
    print("Bienvenidos a JAS Adventures, donde no solo viajas… vives cada destino\n")

    ciudades = [item["ciudad"] for item in base_de_informacion()]
    print("Ciudades disponibles:")
    for ciudad in ciudades:
        print(f"- {ciudad}")

    base_semantica = construir_base_semantica()
    pregunta = ""

    while pregunta.lower() != "salir":
        pregunta = input("\n¿Qué deseas consultar?\n")

        if pregunta.lower() == "salir":
            print("\nHasta luego.")
            break

        resultados = buscar_chunks_relevantes(pregunta, base_semantica, top_k=5)
        contexto = construir_contexto(resultados)
        prompt = construir_prompt(contexto, pregunta)
        respuesta = generar_respuesta(prompt)

        print("\nRespuesta del asistente:")
        print(respuesta)

if __name__ == "__main__":
    main()
