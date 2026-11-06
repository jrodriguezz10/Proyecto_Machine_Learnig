/*
  ========================================================================
  PROYECTO: Monitor de Calidad del Aire & Alarma de Fuego (IoT + ML)
  CURSO   : Machine Learning - UPAO Ciclo 8
  MODO    : ESP32 ACCESS POINT (Wi-Fi "ML") + SONIDO BUZZER CORREGIDO
  ========================================================================

  CONEXIONES FÍSICAS EN ESP32:
  1. Sensor Calidad de Aire (Foto):
     - VCC  -> 3V3 del ESP32
     - GND  -> GND del ESP32
     - AOUT -> GPIO 34 (Pin D34)
     - DOUT -> (Sin conectar)

  2. Sensor DHT11 (Azul):
     - VCC / (+)      -> 3V3 del ESP32
     - GND / (-)      -> GND del ESP32
     - OUT / DATA / S -> GPIO 27 (Pin D27)

  3. Actuadores:
     - LED Verde (Seguro) : Pin D5  (GPIO 5 con resistencia 220Ω, pata corta a GND)
     - LED Rojo (Alarma)  : Pin D4  (GPIO 4 con resistencia 220Ω, pata corta a GND)
     - Buzzer             : Pin D18 (GPIO 18 al positivo (+), negativo (-) a GND)
*/

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <DHT.h>

// ========================================================================
// 1. CONFIGURACIÓN DEL WI-FI EMITIDO POR EL ESP32
// ========================================================================
const char* ap_ssid     = "ML";
const char* ap_password = "12345678";

WebServer server(80);

// ========================================================================
// 2. DEFINICIÓN DE PINES
// ========================================================================
#define PIN_SENSOR_AIRE  34   // Pin D34 (ADC1)
#define PIN_DHT          27   // Pin D27 (DATA)
#define DHTTYPE          DHT11

const int PIN_LED_NORMAL = 5;    // Pin D5 (LED Verde)
const int PIN_LED_ALARM  = 4;    // Pin D4 (LED Rojo)
const int PIN_BUZZER     = 18;   // Pin D18 (Buzzer)

DHT dht(PIN_DHT, DHTTYPE);

// Variables de sensores
float temp = 24.0;
float hum  = 50.0;
int   rawAire = 380;
float factorHumo = 0.0;
float tvoc = 10.0;
float pm25 = 0.05;
float pm10 = 0.04;
float eco2 = 400.0;

int estadoAlarma = 0; // 0 = Seguro, 1 = Fuego

// Función para activar el buzzer (compatible con buzzers activos Y pasivos)
void sonarAlarmaBuzzer(bool encender) {
  if (encender) {
    // Genera tono audible a 2400 Hz para buzzer pasivo o activo
    tone(PIN_BUZZER, 2400);
    digitalWrite(PIN_BUZZER, HIGH);
  } else {
    noTone(PIN_BUZZER);
    digitalWrite(PIN_BUZZER, LOW);
  }
}

// ========================================================================
// 3. RUTAS WEB EN EL ESP32 (192.168.4.1)
// ========================================================================
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>ESP32 ML Sensor</title>";
  html += "<style>body{font-family:sans-serif;background:#f8fafc;color:#0f172a;padding:24px;text-align:center;}";
  html += ".box{background:#fff;border-radius:12px;padding:20px;max-width:380px;margin:20px auto;border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.05);}";
  html += "</style></head><body>";
  html += "<h2>Monitor de Calidad de Aire - UPAO</h2>";
  html += "<div class='box'>";
  html += "<p><strong>Temperatura:</strong> " + String(temp, 1) + " °C</p>";
  html += "<p><strong>Humedad:</strong> " + String(hum, 1) + " %</p>";
  html += "<p><strong>Sensor Aire (AOUT):</strong> " + String(rawAire) + "</p>";
  html += "<p><strong>Estado:</strong> " + String(estadoAlarma == 1 ? "<span style='color:#dc2626;font-weight:bold;'>PELIGRO HUMO</span>" : "<span style='color:#059669;font-weight:bold;'>AIRE NORMAL</span>") + "</p>";
  html += "</div>";
  html += "<p>Abre el Dashboard en tu PC en: <strong>http://localhost:8000</strong></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleSensors() {
  String json = "{";
  json += "\"temperature\":" + String(temp, 2) + ",";
  json += "\"humidity\":" + String(hum, 2) + ",";
  json += "\"tvoc\":" + String(tvoc, 1) + ",";
  json += "\"eco2\":" + String(eco2, 1) + ",";
  json += "\"raw_h2\":" + String(12400.0 + (factorHumo * 800.0), 1) + ",";
  json += "\"raw_ethanol\":" + String(18600.0 + (factorHumo * 1200.0), 1) + ",";
  json += "\"pressure\":939.5,";
  json += "\"pm1_0\":" + String(pm10, 3) + ",";
  json += "\"pm2_5\":" + String(pm25, 3) + ",";
  json += "\"nc0_5\":" + String(0.1 + (factorHumo * 25.0), 3) + ",";
  json += "\"nc1_0\":" + String(0.02 + (factorHumo * 4.0), 3) + ",";
  json += "\"nc2_5\":" + String(0.005 + (factorHumo * 0.1), 3) + ",";
  json += "\"raw_aire\":" + String(rawAire) + ",";
  json += "\"device_id\":\"ESP32_AP_ML\"";
  json += "}";
  server.send(200, "application/json", json);
}

void handleSetAlarm() {
  if (server.hasArg("fire")) {
    estadoAlarma = server.arg("fire").toInt();
    if (estadoAlarma == 1) {
      Serial.println(">>> ALERTA DE FUEGO: Activando LED Rojo y Buzzer <<<");
      digitalWrite(PIN_LED_NORMAL, LOW);   // Apagar verde
      digitalWrite(PIN_LED_ALARM, HIGH);   // Encender rojo
      sonarAlarmaBuzzer(true);             // Sonar buzzer con tono
    } else {
      digitalWrite(PIN_LED_NORMAL, HIGH);  // Encender verde
      digitalWrite(PIN_LED_ALARM, LOW);    // Apagar rojo
      sonarAlarmaBuzzer(false);            // Silenciar buzzer
    }
  }
  server.send(200, "text/plain", "OK");
}

// ========================================================================
// 4. SETUP
// ========================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n========================================================");
  Serial.println("   ESP32 INICIANDO (WI-FI: ML | IP: 192.168.4.1)");
  Serial.println("========================================================");

  pinMode(PIN_LED_NORMAL, OUTPUT);
  pinMode(PIN_LED_ALARM, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  // Encender LED Verde al arrancar
  digitalWrite(PIN_LED_NORMAL, HIGH);

  // Bip de prueba inicial del Buzzer para confirmar que suena físicamente
  sonarAlarmaBuzzer(true);
  delay(250);
  sonarAlarmaBuzzer(false);

  analogReadResolution(12);
  dht.begin();

  // Iniciar Access Point
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ap_ssid, ap_password);

  Serial.println("[OK] Red Wi-Fi 'ML' activa.");
  Serial.print("[*] IP del ESP32: ");
  Serial.println(WiFi.softAPIP());

  server.on("/", handleRoot);
  server.on("/api/sensors", handleSensors);
  server.on("/set_alarm", handleSetAlarm);
  server.begin();
}

unsigned long ultimaLectura = 0;

void loop() {
  server.handleClient();

  if (millis() - ultimaLectura > 1000) {
    ultimaLectura = millis();

    // 1. Leer DHT11
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) temp = t;
    if (!isnan(h)) hum  = h;

    // 2. Leer Sensor de Aire (Foto)
    rawAire = analogRead(PIN_SENSOR_AIRE);

    // 3. Calibración del umbral de humo:
    // Tu sensor en reposo caliente mide ~1100-1300.
    // Solo cuando se le echa humo supera los 1550 y se dispara.
    factorHumo = constrain((float)(rawAire - 1500) / 1600.0, 0.0, 1.0);
    tvoc = 10.0 + (factorHumo * 1400.0);
    pm25 = 0.05 + (factorHumo * 4.2);
    pm10 = 0.04 + (factorHumo * 3.8);
    eco2 = 400.0 + (factorHumo * 450.0);

    Serial.printf("[SENSOR REAL] Temp: %.1f C | Hum: %.1f %% | Aire AOUT: %d | FactorHumo: %.2f\n",
                  temp, hum, rawAire, factorHumo);

    // 4. Enviar a la PC por HTTP POST
    if (WiFi.softAPgetStationNum() > 0) {
      HTTPClient http;
      http.begin("http://192.168.4.2:8000/api/predict");
      http.addHeader("Content-Type", "application/json");

      String jsonPayload = "{";
      jsonPayload += "\"temperature\":" + String(temp, 2) + ",";
      jsonPayload += "\"humidity\":" + String(hum, 2) + ",";
      jsonPayload += "\"tvoc\":" + String(tvoc, 1) + ",";
      jsonPayload += "\"eco2\":" + String(eco2, 1) + ",";
      jsonPayload += "\"raw_h2\":" + String(12400.0 + (factorHumo * 800.0), 1) + ",";
      jsonPayload += "\"raw_ethanol\":" + String(18600.0 + (factorHumo * 1200.0), 1) + ",";
      jsonPayload += "\"pressure\":939.5,";
      jsonPayload += "\"pm1_0\":" + String(pm10, 3) + ",";
      jsonPayload += "\"pm2_5\":" + String(pm25, 3) + ",";
      jsonPayload += "\"nc0_5\":" + String(0.1 + (factorHumo * 25.0), 3) + ",";
      jsonPayload += "\"nc1_0\":" + String(0.02 + (factorHumo * 4.0), 3) + ",";
      jsonPayload += "\"nc2_5\":" + String(0.005 + (factorHumo * 0.1), 3) + ",";
      jsonPayload += "\"raw_aire\":" + String(rawAire) + ",";
      jsonPayload += "\"device_id\":\"ESP32_AP_ML\"";
      jsonPayload += "}";

      int httpCode = http.POST(jsonPayload);
      if (httpCode > 0) {
        String response = http.getString();
        if (response.indexOf("\"fire_alarm\":1") != -1 || response.indexOf("\"fire_alarm\": 1") != -1) {
          digitalWrite(PIN_LED_NORMAL, LOW);
          digitalWrite(PIN_LED_ALARM, HIGH);
          sonarAlarmaBuzzer(true);
        } else {
          digitalWrite(PIN_LED_NORMAL, HIGH);
          digitalWrite(PIN_LED_ALARM, LOW);
          sonarAlarmaBuzzer(false);
        }
      }
      http.end();
    }
  }
}
