import requests
import unicodedata


def get_weather(city: str) -> str:
    """
    Récupère la météo réelle d'une ville via wttr.in (gratuit, sans clé API).
    """
    try:
        # Normaliser les accents pour wttr.in (ex: Bogotá → Bogota)
        city_ascii = unicodedata.normalize('NFD', city).encode('ascii', 'ignore').decode('utf-8')
        url = f"https://wttr.in/{city_ascii}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extraire les données météo
        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        city_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        description = current["weatherDesc"][0]["value"]
        wind_kmph = current["windspeedKmph"]

        return (
            f"🌤️ Météo à {city_name}, {country} :\n"
            f"- Température : {temp_c}°C (ressenti {feels_like}°C)\n"
            f"- Conditions : {description}\n"
            f"- Humidité : {humidity}%\n"
            f"- Vent : {wind_kmph} km/h"
        )

    except requests.exceptions.ConnectionError:
        return f"❌ Impossible de se connecter au service météo pour '{city}'."
    except requests.exceptions.Timeout:
        return f"❌ Le service météo n'a pas répondu pour '{city}'."
    except (KeyError, IndexError):
        return f"❌ Ville '{city}' introuvable. Essaie avec un nom de ville en anglais."
    except Exception as e:
        return f"❌ Erreur météo : {e}"


if __name__ == "__main__":
    print(get_weather("Paris"))
    print(get_weather("Dakar"))
    print(get_weather("Bogotá"))