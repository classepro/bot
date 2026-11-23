# Configuration Streamlit

- `enableAgentApi = true` : permet de piloter l'app par API JSON (tests automatisés des 7 pages).
- `address = 0.0.0.0` : nécessaire pour l'aperçu live dans un conteneur.
- `enableCORS = false` / `enableXsrfProtection = false` : uniquement pour un environnement
  de démonstration derrière un proxy. **En production**, retirez ces deux lignes.
