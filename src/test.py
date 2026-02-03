from google import genai

# initialise le client avec ta clé API
client = genai.Client(api_key="AIzaSyDWXKyDN6nHg1Kwkxmq6AbpU8u_VgOjkQ4")  # remplace par ta vraie clé

response = client.models.generate_content(
    model="gemini-2.5-flash",          # modèle de génération
    contents="Écris un court conseil de carrière pour une personne avec des compétences en data et ML."
)

print("Texte généré :", response.text)
