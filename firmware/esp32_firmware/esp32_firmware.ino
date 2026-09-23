#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// =====================================================
// REFLECTAI (AAINA) — ESP32 FIRMWARE
//
// Hardware:
//   - ESP32 NodeMCU / Dev Module (WROOM-32)
//   - DHT22 (or DHT11) Temperature & Humidity Sensor
//   - HC-SR501 / AM312 PIR Motion Sensor
//   - LDR Light Sensor Module
//   - 0.96" I2C OLED Display (SSD1306, 128x64)
//   - Piezo Buzzer (Passive or Active)
//   - 4 Push Buttons (User 1, User 2, Guest, Privacy)
//
// Required Arduino IDE Libraries:
//   1. "Adafruit SSD1306" by Adafruit
//   2. "Adafruit GFX Library" by Adafruit
//   3. "DHT sensor library" by Adafruit
//   4. "Adafruit Unified Sensor" by Adafruit
// =====================================================

// =====================================================
// WIFI / RASPBERRY PI CONFIGURATION
// =====================================================
const char* WIFI_SSID     = "Mithi papdi";
const char* WIFI_PASSWORD = "9468665211";

// Target Raspberry Pi / Dashboard Backend API
// Update IP if your Pi or PC IP changes
const char* PI_STATE_URL =
  "http://192.168.173.9:5000/api/esp32/state";

// Security shared token matching backend
const char* PI_SHARED_TOKEN =
  "KJwprEKT7YMu0WEYn0lcAJIZaNq0-m7tYr7Fx1vl9q8";

// =====================================================
// PIN DEFINITIONS
// =====================================================
#define DHT_PIN      4     // DHT22 Data Pin
#define DHT_TYPE     DHT22 // Change to DHT11 if using a DHT11 sensor
#define PIR_PIN      27    // PIR Motion Sensor
#define LDR_PIN      34    // LDR Light Sensor (ADC input)

#define USER1_PIN    13    // Profile 1 Button (Maanik) -> Active LOW (GND)
#define USER2_PIN    25    // Profile 2 Button (Ayush)  -> Active LOW (GND)
#define GUEST_PIN    14    // Guest Profile Button      -> Active LOW (GND)
#define PRIVACY_PIN  26    // Privacy Toggle Button     -> Active LOW (GND)

#define BUZZER_PIN   18    // Piezo Buzzer Pin

// =====================================================
// OLED CONFIGURATION (I2C SDA=21, SCL=22)
// =====================================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1
#define OLED_ADDRESS 0x3C  // Default I2C address (auto-fallbacks to 0x3D)

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
DHT dht(DHT_PIN, DHT_TYPE);

// =====================================================
// SYSTEM STATE
// =====================================================
String currentUser = "Guest";
bool privacyMode   = false;

bool user1LastState   = HIGH;
bool user2LastState   = HIGH;
bool guestLastState   = HIGH;
bool privacyLastState = HIGH;

unsigned long lastButtonTime[4] = {0, 0, 0, 0};
const unsigned long DEBOUNCE_DELAY = 220; // 220ms debounce

float temperature = NAN;
float humidity    = NAN;
int   lightState  = LOW;
int   motionState = LOW;
bool  motionLatch = false; // Latches fast motion pulses until sent

unsigned long lastDHTRead     = 0;
unsigned long lastServerSend  = 0;
unsigned long lastOLEDUpdate  = 0;
int           serverFailCount = 0;
int           dhtFailCount    = 0;

const unsigned long DHT_INTERVAL    = 2500; // Read DHT every 2.5s
const unsigned long SERVER_INTERVAL = 3000; // Telemetry post every 3s
const unsigned long OLED_INTERVAL   = 400;  // Update OLED every 400ms

// =====================================================
// FORWARD DECLARATIONS
// =====================================================
void beep(unsigned int durationMs = 80);
void connectWiFi();
void readDHT();
void readSensors();
void updateOLED();
void sendToPi(bool forceImmediate);
void sendToPi();
void handleButtons();

// =====================================================
// BUZZER FEEDBACK
// (Clean tone generator - 100% compatible across all ESP32 cores)
// =====================================================
void beep(unsigned int durationMs) {
  pinMode(BUZZER_PIN, OUTPUT);
  unsigned long start = millis();
  while (millis() - start < durationMs) {
    digitalWrite(BUZZER_PIN, HIGH);
    delayMicroseconds(250); // 2 kHz tone
    digitalWrite(BUZZER_PIN, LOW);
    delayMicroseconds(250);
  }
  digitalWrite(BUZZER_PIN, LOW);
}

// =====================================================
// WIFI CONNECTION
// =====================================================
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println();
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(300);
    Serial.print(".");
    attempts++;
  }

  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("[OK] Wi-Fi Connected!");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.print("[WARN] Wi-Fi Connection failed (status: ");
    Serial.print(WiFi.status());
    Serial.println("). Ensure hotspot is 2.4 GHz.");
  }
}

// =====================================================
// SENSORS
// =====================================================
void readDHT() {
  if (millis() - lastDHTRead < DHT_INTERVAL) {
    return;
  }
  lastDHTRead = millis();

  float t = dht.readTemperature();
  float h = dht.readHumidity();

  if (!isnan(t) && !isnan(h)) {
    temperature = t;
    humidity = h;
    dhtFailCount = 0;
    Serial.print("Temp: ");
    Serial.print(temperature, 1);
    Serial.print(" C | Hum: ");
    Serial.print(humidity, 1);
    Serial.println(" %");
  } else {
    dhtFailCount++;
    if (dhtFailCount == 1 || dhtFailCount % 10 == 0) {
      Serial.print("[WARN] DHT22 Read Warning (NaN attempt ");
      Serial.print(dhtFailCount);
      Serial.println("). Keeping previous reading.");
    }
  }
}

void readSensors() {
  // LDR Sensor (Digital DO comparator module or bare resistor divider)
  lightState = digitalRead(LDR_PIN);

  // PIR Motion Sensor
  int pir = digitalRead(PIR_PIN);
  if (pir == HIGH) {
    motionState = HIGH;
    motionLatch = true; // Latch motion until next telemetry packet
  } else {
    motionState = LOW;
  }
}

// =====================================================
// OLED DISPLAY
// =====================================================
void updateOLED() {
  if (millis() - lastOLEDUpdate < OLED_INTERVAL) {
    return;
  }
  lastOLEDUpdate = millis();

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // Header Banner
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("REFLECTAI");

  // WiFi status indicator in header
  display.setCursor(85, 0);
  if (WiFi.status() == WL_CONNECTED) {
    display.print("[WiFi]");
  } else {
    display.print("[NoNet]");
  }
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (privacyMode) {
    display.setTextSize(2);
    display.setCursor(18, 22);
    display.println("PRIVATE");

    display.setTextSize(1);
    display.setCursor(12, 48);
    display.println("Sensors & Cam Muted");
  } else {
    // Current User
    display.setTextSize(1);
    display.setCursor(0, 14);
    display.print("User: ");
    display.println(currentUser);

    // Temperature
    display.setCursor(0, 27);
    display.print("Temp: ");
    if (isnan(temperature)) {
      display.println("--.- C");
    } else {
      display.print(temperature, 1);
      display.println(" C");
    }

    // Humidity
    display.setCursor(0, 38);
    display.print("Hum:  ");
    if (isnan(humidity)) {
      display.println("-- %");
    } else {
      display.print(humidity, 0);
      display.println(" %");
    }

    // Light Status
    display.setCursor(0, 50);
    display.print("Light:");
    display.print(lightState == HIGH ? "BRIGHT" : "DARK");

    // Motion Status
    display.setCursor(76, 50);
    display.print((motionState == HIGH || motionLatch) ? "MOTION" : "IDLE");
  }

  display.display();
}

// =====================================================
// SEND DATA TO RASPBERRY PI
// =====================================================
void sendToPi() {
  sendToPi(false);
}

void sendToPi(bool forceImmediate) {
  // If not forced, check interval
  if (!forceImmediate && (millis() - lastServerSend < SERVER_INTERVAL)) {
    return;
  }

  // If Pi was unreachable, apply 8s backoff so main loop never stutters
  if (!forceImmediate && serverFailCount > 0 && (millis() - lastServerSend < 8000)) {
    return;
  }
  lastServerSend = millis();

  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastReconnectAttempt = 0;
    if (millis() - lastReconnectAttempt > 8000) {
      lastReconnectAttempt = millis();
      Serial.println("[WiFi] Not connected. Attempting background reconnect...");
      WiFi.reconnect();
    }
    return;
  }

  String lightStatus = (lightState == HIGH) ? "bright" : "dark";
  bool sendMotion = (motionState == HIGH) || motionLatch;
  motionLatch = false; // Clear latch upon transmission

  // Build clean JSON packet
  String json = "{";
  json += "\"device\":\"esp32\",";
  json += "\"status\":\"online\",";

  if (isnan(temperature)) {
    json += "\"temperature\":null,";
  } else {
    json += "\"temperature\":" + String(temperature, 1) + ",";
  }

  if (isnan(humidity)) {
    json += "\"humidity\":null,";
  } else {
    json += "\"humidity\":" + String(humidity, 1) + ",";
  }

  json += "\"light\":\"" + lightStatus + "\",";
  json += "\"motion\":" + String(sendMotion ? "true" : "false") + ",";
  json += "\"user\":\"" + currentUser + "\",";
  json += "\"privacy\":" + String(privacyMode ? "true" : "false");
  json += "}";

  WiFiClient client;
  HTTPClient http;

  if (!http.begin(client, PI_STATE_URL)) {
    Serial.println("[ERROR] Unable to begin HTTPClient to Pi");
    return;
  }

  http.setTimeout(2000); // 2 second timeout so loop never hangs
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-ReflectAI-Token", PI_SHARED_TOKEN);

  int httpCode = http.POST(json);

  if (httpCode > 0) {
    serverFailCount = 0;
    Serial.print("HTTP ");
    Serial.print(httpCode);
    Serial.print(" -> ");
    String response = http.getString();
    Serial.println(response);
  } else {
    serverFailCount++;
    Serial.print("[HTTP ERROR] code: ");
    Serial.print(httpCode);
    Serial.print(" (");
    Serial.print(HTTPClient::errorToString(httpCode));
    Serial.print("). Check Pi server at ");
    Serial.println(PI_STATE_URL);
  }

  http.end();
  client.stop(); // Cleanly close socket
}

// =====================================================
// BUTTON HANDLING (Hardware Debounced + Instant Trigger)
// =====================================================
void handleButtons() {
  unsigned long now = millis();

  // USER 1 (Profile 1 - Maanik)
  bool u1 = digitalRead(USER1_PIN);
  if (u1 == LOW && user1LastState == HIGH && (now - lastButtonTime[0] > DEBOUNCE_DELAY)) {
    lastButtonTime[0] = now;
    currentUser = "User 1";
    privacyMode = false;
    Serial.println(">>> USER 1 (Profile 1) SELECTED <<<");
    beep();
    updateOLED();
    sendToPi(true);
  }
  user1LastState = u1;

  // USER 2 (Profile 2 - Ayush)
  bool u2 = digitalRead(USER2_PIN);
  if (u2 == LOW && user2LastState == HIGH && (now - lastButtonTime[1] > DEBOUNCE_DELAY)) {
    lastButtonTime[1] = now;
    currentUser = "User 2";
    privacyMode = false;
    Serial.println(">>> USER 2 (Profile 2) SELECTED <<<");
    beep();
    updateOLED();
    sendToPi(true);
  }
  user2LastState = u2;

  // GUEST PROFILE
  bool g = digitalRead(GUEST_PIN);
  if (g == LOW && guestLastState == HIGH && (now - lastButtonTime[2] > DEBOUNCE_DELAY)) {
    lastButtonTime[2] = now;
    currentUser = "Guest";
    privacyMode = false;
    Serial.println(">>> GUEST SELECTED <<<");
    beep();
    updateOLED();
    sendToPi(true);
  }
  guestLastState = g;

  // PRIVACY TOGGLE
  bool p = digitalRead(PRIVACY_PIN);
  if (p == LOW && privacyLastState == HIGH && (now - lastButtonTime[3] > DEBOUNCE_DELAY)) {
    lastButtonTime[3] = now;
    privacyMode = !privacyMode;
    Serial.print(">>> PRIVACY MODE: ");
    Serial.println(privacyMode ? "ON <<<" : "OFF <<<");
    beep();
    updateOLED();
    sendToPi(true);
  }
  privacyLastState = p;
}

// =====================================================
// ARDUINO SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n================================");
  Serial.println(" REFLECTAI - AAINA (ESP32)");
  Serial.println("================================");

  // I2C for SSD1306 OLED (SDA=21, SCL=22)
  Wire.begin(21, 22);

  // Initialize DHT22
  dht.begin();

  // Pin modes
  pinMode(LDR_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);

  pinMode(USER1_PIN, INPUT_PULLUP);
  pinMode(USER2_PIN, INPUT_PULLUP);
  pinMode(GUEST_PIN, INPUT_PULLUP);
  pinMode(PRIVACY_PIN, INPUT_PULLUP);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // Initialize OLED Display with automatic address fallback (0x3C -> 0x3D)
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
      Serial.println("[ERROR] OLED Display Init Failed! Check I2C wiring (SDA=21, SCL=22).");
    } else {
      Serial.println("[OK] OLED Display Ready at address 0x3D.");
    }
  } else {
    Serial.println("[OK] OLED Display Ready at address 0x3C.");
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(10, 22);
  display.println("REFLECTAI");
  display.display();
  delay(1000);

  // Connect to Wi-Fi
  connectWiFi();

  beep();
  Serial.println("[READY] System initialized. Entering main loop.\n");
}

// =====================================================
// ARDUINO MAIN LOOP
// =====================================================
void loop() {
  handleButtons(); // Checks button presses & triggers instant Pi sync
  readDHT();       // Periodic DHT read (every 2.5s)
  readSensors();   // Digital LDR and PIR reads
  updateOLED();    // Periodic OLED refresh (every 400ms)
  sendToPi();      // Periodic sensor telemetry (every 3s)

  delay(20);
}
