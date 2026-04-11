## Installation

### 2. Installer les dépendances

pip install -r requirements.txt

### 3. Configurer la clé API Mistral

Crée un fichier `.env` à la racine du projet en copiant le template :

cp .env.example .env

Puis remplis `.env` avec ta clé :

MISTRAL_API_KEY=ta-clé-mistral

Pour obtenir une clé Mistral gratuite :
1. Crée un compte sur https://console.mistral.ai
2. Va dans API Keys → Create new key
3. Copie la clé et colle-la dans ton `.env`

⚠️ Ne commite jamais le fichier `.env` sur GitHub. Il est déjà dans le `.gitignore`.