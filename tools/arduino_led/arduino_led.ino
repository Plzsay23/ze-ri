#include <SoftwareSerial.h>

// HC-05 / ZS-040
// HC-05 TXD -> Arduino D10
// HC-05 RXD <- Arduino D11
SoftwareSerial BT(10, 11); // RX, TX

// Relay pins
const int RELAY_R = 3;
const int RELAY_G = 4;
const int RELAY_B = 5;

// 대부분 5V 릴레이 모듈은 LOW trigger
const bool RELAY_ACTIVE_LOW = true;

bool stateR = false;
bool stateG = false;
bool stateB = false;

void relayWrite(int pin, bool on) {
  if (RELAY_ACTIVE_LOW) {
    digitalWrite(pin, on ? LOW : HIGH);
  } else {
    digitalWrite(pin, on ? HIGH : LOW);
  }
}

void applyRGB(bool r, bool g, bool b) {
  stateR = r;
  stateG = g;
  stateB = b;

  relayWrite(RELAY_R, stateR);
  relayWrite(RELAY_G, stateG);
  relayWrite(RELAY_B, stateB);
}

void printState(const char* src, const char* name) {
  Serial.print("[");
  Serial.print(src);
  Serial.print("] ");
  Serial.print(name);
  Serial.print("  RGB=");
  Serial.print(stateR ? "1" : "0");
  Serial.print(stateG ? "1" : "0");
  Serial.println(stateB ? "1" : "0");

  BT.print("OK ");
  BT.print(name);
  BT.print(" RGB=");
  BT.print(stateR ? "1" : "0");
  BT.print(stateG ? "1" : "0");
  BT.println(stateB ? "1" : "0");
}

void handleCommand(char c, const char* src) {
  if (c == '\n' || c == '\r' || c == ' ') {
    return;
  }

  Serial.print("[");
  Serial.print(src);
  Serial.print("] received: ");
  Serial.println(c);

  switch (c) {
    case '0':
      applyRGB(false, false, false);
      printState(src, "OFF");
      break;

    case '1':
      applyRGB(true, false, false);
      printState(src, "RED");
      break;

    case '2':
      applyRGB(false, true, false);
      printState(src, "GREEN");
      break;

    case '3':
      applyRGB(false, false, true);
      printState(src, "BLUE");
      break;

    case '4':
      applyRGB(true, true, false);
      printState(src, "YELLOW");
      break;

    case '5':
      applyRGB(true, false, true);
      printState(src, "MAGENTA");
      break;

    case '6':
      applyRGB(false, true, true);
      printState(src, "CYAN");
      break;

    case '7':
      applyRGB(true, true, true);
      printState(src, "WHITE");
      break;

    // 문자 명령도 같이 지원
    case 'r':
    case 'R':
      applyRGB(true, false, false);
      printState(src, "RED");
      break;

    case 'g':
    case 'G':
      applyRGB(false, true, false);
      printState(src, "GREEN");
      break;

    case 'b':
    case 'B':
      applyRGB(false, false, true);
      printState(src, "BLUE");
      break;

    case 'y':
    case 'Y':
      applyRGB(true, true, false);
      printState(src, "YELLOW");
      break;

    case 'm':
    case 'M':
      applyRGB(true, false, true);
      printState(src, "MAGENTA");
      break;

    case 'c':
    case 'C':
      applyRGB(false, true, true);
      printState(src, "CYAN");
      break;

    case 'w':
    case 'W':
    case 'a':
    case 'A':
      applyRGB(true, true, true);
      printState(src, "WHITE");
      break;

    case 'o':
    case 'O':
      applyRGB(false, false, false);
      printState(src, "OFF");
      break;

    default:
      Serial.print("[");
      Serial.print(src);
      Serial.print("] unknown command: ");
      Serial.println(c);

      BT.print("ERR UNKNOWN_CMD ");
      BT.println(c);
      break;
  }
}

void setup() {
  // LOW trigger 릴레이 부팅 순간 오동작 방지
  digitalWrite(RELAY_R, HIGH);
  digitalWrite(RELAY_G, HIGH);
  digitalWrite(RELAY_B, HIGH);

  pinMode(RELAY_R, OUTPUT);
  pinMode(RELAY_G, OUTPUT);
  pinMode(RELAY_B, OUTPUT);

  applyRGB(false, false, false);

  Serial.begin(9600);
  BT.begin(9600);

  delay(300);

  Serial.println("READY RGB 8-COLOR RELAY CONTROLLER");
  Serial.println("Commands:");
  Serial.println("0=OFF, 1=RED, 2=GREEN, 3=BLUE");
  Serial.println("4=YELLOW, 5=MAGENTA, 6=CYAN, 7=WHITE");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    handleCommand(c, "USB");
  }

  while (BT.available()) {
    char c = BT.read();
    handleCommand(c, "BT");
  }
}