#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// =====================================================
// REFLECTAI (AAINA)
// COMPLETE ESP32 FIRMWARE
// =====================================================

// =====================================================
// WIFI / RASPBERRY PI
// =====================================================

const char* WIFI_SSID = "Airtel_AP Wifi";
const char* WIFI_PASSWORD = "Sdr@3232";

const char* PI_STATE_URL =
  "http://192.168.1.11:5000/api/esp32/state";

// Use the SAME token that is currently working
const char* PI_SHARED_TOKEN =
  "KJwprEKT7YMu0WEYn0lcAJIZaNq0-m7tYr7Fx1vl9q8";

// =====================================================
// DHT22
// =====================================================

#define DHT_PIN 4
#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

// =====================================================
// PIR
// =====================================================

#define PIR_PIN 27

// =====================================================
// LDR
// =====================================================

#define LDR_PIN 34

// =====================================================
// OLED
// =====================================================

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET
);

// =====================================================
// BUTTONS
// =====================================================

#define USER1_PIN    13
#define USER2_PIN    25
#define GUEST_PIN    14
#define PRIVACY_PIN  26

// =====================================================
// BUZZER
// =====================================================

#define BUZZER_PIN 18

// =====================================================
// SYSTEM STATE
// =====================================================

String currentUser = "Guest";

bool privacyMode = false;

bool user1LastState = HIGH;
bool user2LastState = HIGH;
bool guestLastState = HIGH;
bool privacyLastState = HIGH;

// =====================================================
// SENSOR VALUES
// =====================================================

float temperature = NAN;
float humidity = NAN;

int lightState = LOW;
int motionState = LOW;

// =====================================================
// TIMERS
// =====================================================

unsigned long lastDHTRead = 0;
unsigned long lastServerSend = 0;
unsigned long lastOLEDUpdate = 0;

const unsigned long DHT_INTERVAL = 2500;
const unsigned long SERVER_INTERVAL = 3000;
const unsigned long OLED_INTERVAL = 500;

// =====================================================
// BUZZER
// =====================================================

void beep() {

  tone(BUZZER_PIN, 2000);

  delay(150);

  noTone(BUZZER_PIN);
}

// =====================================================
// WIFI CONNECTION
// =====================================================

void connectWiFi() {

  Serial.println();
  Serial.println("Connecting to Wi-Fi...");

  WiFi.mode(WIFI_STA);

  WiFi.begin(
    WIFI_SSID,
    WIFI_PASSWORD
  );

  int attempts = 0;

  while (
    WiFi.status() != WL_CONNECTED &&
    attempts < 40
  ) {

    delay(500);

    Serial.print(".");

    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println("Wi-Fi CONNECTED");

    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());

  } else {

    Serial.println("Wi-Fi CONNECTION FAILED");
  }
}

// =====================================================
// DHT22
// =====================================================

void readDHT() {

  if (
    millis() - lastDHTRead <
    DHT_INTERVAL
  ) {

    return;
  }

  lastDHTRead = millis();

  float newTemperature =
    dht.readTemperature();

  float newHumidity =
    dht.readHumidity();

  if (
    !isnan(newTemperature) &&
    !isnan(newHumidity)
  ) {

    temperature = newTemperature;
    humidity = newHumidity;

    Serial.print("Temperature: ");
    Serial.print(temperature, 1);

    Serial.print(" C | Humidity: ");
    Serial.print(humidity, 1);

    Serial.println(" %");

  } else {

    Serial.println(
      "DHT22: READ ERROR"
    );
  }
}

// =====================================================
// READ DIGITAL SENSORS
// =====================================================

void readSensors() {

  lightState =
    digitalRead(LDR_PIN);

  motionState =
    digitalRead(PIR_PIN);
}

// =====================================================
// OLED
// =====================================================

void updateOLED() {

  if (
    millis() - lastOLEDUpdate <
    OLED_INTERVAL
  ) {

    return;
  }

  lastOLEDUpdate = millis();

  display.clearDisplay();

  display.setTextColor(
    SSD1306_WHITE
  );

  // ---------------------------------------------------
  // HEADER
  // ---------------------------------------------------

  display.setTextSize(1);

  display.setCursor(0, 0);

  display.println(
    "REFLECTAI - AAINA"
  );

  display.drawLine(
    0,
    10,
    127,
    10,
    SSD1306_WHITE
  );

  // ---------------------------------------------------
  // PRIVACY MODE
  // ---------------------------------------------------

  if (privacyMode) {

    display.setTextSize(2);

    display.setCursor(18, 22);

    display.println(
      "PRIVATE"
    );

    display.setTextSize(1);

    display.setCursor(25, 48);

    display.println(
      "Display Protected"
    );

  } else {

    // -------------------------------------------------
    // USER
    // -------------------------------------------------

    display.setTextSize(1);

    display.setCursor(0, 14);

    display.print("User: ");

    display.println(
      currentUser
    );

    // -------------------------------------------------
    // TEMPERATURE
    // -------------------------------------------------

    display.setCursor(0, 27);

    display.print("Temp: ");

    if (isnan(temperature)) {

      display.println("ERROR");

    } else {

      display.print(
        temperature,
        1
      );

      display.println(" C");
    }

    // -------------------------------------------------
    // HUMIDITY
    // -------------------------------------------------

    display.setCursor(0, 38);

    display.print("Hum: ");

    if (isnan(humidity)) {

      display.println("ERROR");

    } else {

      display.print(
        humidity,
        0
      );

      display.println(" %");
    }

    // -------------------------------------------------
    // LIGHT
    // -------------------------------------------------

    display.setCursor(0, 50);

    display.print("Light:");

    if (lightState == HIGH) {

      display.print(
        "BRIGHT"
      );

    } else {

      display.print(
        "DARK"
      );
    }

    // -------------------------------------------------
    // MOTION
    // -------------------------------------------------

    display.setCursor(76, 50);

    if (motionState == HIGH) {

      display.print(
        "MOTION"
      );

    } else {

      display.print(
        "NONE"
      );
    }
  }

  display.display();
}

// =====================================================
// SEND DATA TO RASPBERRY PI
// =====================================================

void sendToPi() {

  if (
    millis() - lastServerSend <
    SERVER_INTERVAL
  ) {

    return;
  }

  lastServerSend = millis();

  if (
    WiFi.status() != WL_CONNECTED
  ) {

    Serial.println(
      "Wi-Fi disconnected"
    );

    connectWiFi();

    return;
  }

  // ---------------------------------------------------
  // SENSOR VALUES
  // ---------------------------------------------------

  String lightStatus;

  if (lightState == HIGH) {

    lightStatus = "bright";

  } else {

    lightStatus = "dark";
  }

  // ---------------------------------------------------
  // HTTP
  // ---------------------------------------------------

  HTTPClient http;

  http.begin(
    PI_STATE_URL
  );

  http.addHeader(
    "Content-Type",
    "application/json"
  );

  // IMPORTANT:
  // Raspberry Pi backend expects this header.
  http.addHeader(
    "X-ReflectAI-Token",
    PI_SHARED_TOKEN
  );

  // ---------------------------------------------------
  // JSON
  // ---------------------------------------------------

  String json = "{";

  json += "\"device\":\"esp32\",";
  json += "\"status\":\"online\",";

  if (isnan(temperature)) {

    json += "\"temperature\":null,";

  } else {

    json += "\"temperature\":";
    json += String(
      temperature,
      1
    );
    json += ",";
  }

  if (isnan(humidity)) {

    json += "\"humidity\":null,";

  } else {

    json += "\"humidity\":";
    json += String(
      humidity,
      1
    );
    json += ",";
  }

  json += "\"light\":\"";
  json += lightStatus;
  json += "\",";

  json += "\"motion\":";

  if (motionState == HIGH) {

    json += "true";

  } else {

    json += "false";
  }

  json += ",";

  json += "\"user\":\"";
  json += currentUser;
  json += "\",";

  json += "\"privacy\":";

  if (privacyMode) {

    json += "true";

  } else {

    json += "false";
  }

  json += "}";

  // ---------------------------------------------------
  // SEND
  // ---------------------------------------------------

  Serial.println();
  Serial.println(
    "Sending data to Raspberry Pi..."
  );

  Serial.print(
    "JSON: "
  );

  Serial.println(
    json
  );

  int httpCode =
    http.POST(json);

  Serial.print(
    "HTTP Response: "
  );

  Serial.println(
    httpCode
  );

  if (httpCode > 0) {

    String response =
      http.getString();

    Serial.print(
      "Server Response: "
    );

    Serial.println(
      response
    );
  }

  http.end();
}

// =====================================================
// BUTTON HANDLING
// =====================================================

void handleButtons() {

  // ===================================================
  // USER 1
  // ===================================================

  bool user1State =
    digitalRead(USER1_PIN);

  if (
    user1State == LOW &&
    user1LastState == HIGH
  ) {

    currentUser = "User 1";

    privacyMode = false;

    Serial.println(
      "USER 1 SELECTED"
    );

    beep();
  }

  user1LastState =
    user1State;

  // ===================================================
  // USER 2
  // ===================================================

  bool user2State =
    digitalRead(USER2_PIN);

  if (
    user2State == LOW &&
    user2LastState == HIGH
  ) {

    currentUser = "User 2";

    privacyMode = false;

    Serial.println(
      "USER 2 SELECTED"
    );

    beep();
  }

  user2LastState =
    user2State;

  // ===================================================
  // GUEST
  // ===================================================

  bool guestState =
    digitalRead(GUEST_PIN);

  if (
    guestState == LOW &&
    guestLastState == HIGH
  ) {

    currentUser = "Guest";

    privacyMode = false;

    Serial.println(
      "GUEST SELECTED"
    );

    beep();
  }

  guestLastState =
    guestState;

  // ===================================================
  // PRIVACY
  // ===================================================

  bool privacyState =
    digitalRead(PRIVACY_PIN);

  if (
    privacyState == LOW &&
    privacyLastState == HIGH
  ) {

    privacyMode =
      !privacyMode;

    Serial.print(
      "PRIVACY MODE: "
    );

    if (privacyMode) {

      Serial.println(
        "ON"
      );

    } else {

      Serial.println(
        "OFF"
      );
    }

    beep();
  }

  privacyLastState =
    privacyState;
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(
    115200
  );

  delay(2000);

  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "REFLECTAI - AAINA"
  );

  Serial.println(
    "ESP32 COMPLETE SYSTEM"
  );

  Serial.println(
    "================================"
  );

  // ===================================================
  // I2C
  // ===================================================

  Wire.begin(
    21,
    22
  );

  // ===================================================
  // DHT22
  // ===================================================

  dht.begin();

  // ===================================================
  // SENSOR PINS
  // ===================================================

  pinMode(
    LDR_PIN,
    INPUT
  );

  pinMode(
    PIR_PIN,
    INPUT
  );

  // ===================================================
  // BUTTONS
  // ===================================================

  pinMode(
    USER1_PIN,
    INPUT_PULLUP
  );

  pinMode(
    USER2_PIN,
    INPUT_PULLUP
  );

  pinMode(
    GUEST_PIN,
    INPUT_PULLUP
  );

  pinMode(
    PRIVACY_PIN,
    INPUT_PULLUP
  );

  // ===================================================
  // BUZZER
  // ===================================================

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  noTone(
    BUZZER_PIN
  );

  // ===================================================
  // OLED
  // ===================================================

  if (
    !display.begin(
      SSD1306_SWITCHCAPVCC,
      OLED_ADDRESS
    )
  ) {

    Serial.println(
      "OLED FAILED!"
    );

  } else {

    Serial.println(
      "OLED: OK"
    );

    display.clearDisplay();

    display.setTextColor(
      SSD1306_WHITE
    );

    display.setTextSize(2);

    display.setCursor(
      5,
      20
    );

    display.println(
      "REFLECTAI"
    );

    display.display();

    delay(1500);
  }

  // ===================================================
  // WIFI
  // ===================================================

  connectWiFi();

  // ===================================================
  // STATUS
  // ===================================================

  Serial.println();

  Serial.println(
    "SYSTEM STATUS"
  );

  Serial.println(
    "---------------"
  );

  Serial.println(
    "DHT22   : READY"
  );

  Serial.println(
    "PIR     : READY"
  );

  Serial.println(
    "LDR     : READY"
  );

  Serial.println(
    "OLED    : READY"
  );

  Serial.println(
    "BUTTONS : READY"
  );

  Serial.println(
    "BUZZER  : READY"
  );

  Serial.println(
    "Wi-Fi   : READY"
  );

  Serial.println(
    "---------------"
  );

  beep();
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  handleButtons();

  readDHT();

  readSensors();

  updateOLED();

  sendToPi();

  delay(20);
}
