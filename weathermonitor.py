import requests

class WeatherMonitor:
    def __init__(self):
        self.api_key = 'YOUR_API_KEY'

    def get_weather(self, city, state):
        geocoding_url = f'https://api.openweathermap.org/data/2.5/weather?q={city},{state}&appid={self.api_key}'
        response = requests.get(geocoding_url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None

# Sample usage:
# monitor = WeatherMonitor()
# weather_data = monitor.get_weather('Los Angeles', 'CA')
# print(weather_data)