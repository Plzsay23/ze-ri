#include <Encoder.h>

// ===== Motor pins: confirmed working wiring =====
#define PWMA 4
#define DIRA1 A5
#define DIRA2 A4

#define PWMB 6
#define DIRB1 43
#define DIRB2 42

#define PWMC 11
#define DIRC1 35
#define DIRC2 34

#define PWMD 7
#define DIRD1 37
#define DIRD2 36

// ===== Encoder pins =====
Encoder encA(18, 31);   // actual RF
Encoder encB(19, 38);   // actual LF, inverted
Encoder encC(3, 49);    // actual RR
Encoder encD(2, A1);    // actual LR, inverted

// ===== Speed settings =====
int Motor_PWM = 60;             // max PWM used by V vx vy wz command
const int MIN_MOTION_PWM = 18;  // helps overcome static friction; lower = smoother, too low = stall

const unsigned long ENC_PERIOD_MS = 100;
unsigned long lastEncMs = 0;

const unsigned long CMD_TIMEOUT_MS = 800;
unsigned long lastCmdMs = 0;

// ===== Smooth velocity ramp =====
// V vx vy wz target is not applied instantly. Each wheel command slews toward target.
const unsigned long RAMP_PERIOD_MS = 20;
unsigned long lastRampMs = 0;

// normalized wheel command step per RAMP_PERIOD_MS.
// Smaller = smoother/slower response. Larger = sharper/faster response.
float ACCEL_STEP = 0.030f;
float DECEL_STEP = 0.050f;

float targetLF = 0.0f, targetRF = 0.0f, targetLR = 0.0f, targetRR = 0.0f;
float currentLF = 0.0f, currentRF = 0.0f, currentLR = 0.0f, currentRR = 0.0f;

char lineBuf[96];
uint8_t lineLen = 0;
bool lineMode = false;

// ===== Motor macros: existing directions preserved =====
#define MOTORA_FORWARD(pwm)    do{digitalWrite(DIRA1,HIGH); digitalWrite(DIRA2,LOW);  analogWrite(PWMA,pwm);}while(0)
#define MOTORA_BACKOFF(pwm)    do{digitalWrite(DIRA1,LOW);  digitalWrite(DIRA2,HIGH); analogWrite(PWMA,pwm);}while(0)
#define MOTORA_STOP()          do{digitalWrite(DIRA1,LOW);  digitalWrite(DIRA2,LOW);  analogWrite(PWMA,0);}while(0)

#define MOTORB_FORWARD(pwm)    do{digitalWrite(DIRB1,LOW);  digitalWrite(DIRB2,HIGH); analogWrite(PWMB,pwm);}while(0)
#define MOTORB_BACKOFF(pwm)    do{digitalWrite(DIRB1,HIGH); digitalWrite(DIRB2,LOW);  analogWrite(PWMB,pwm);}while(0)
#define MOTORB_STOP()          do{digitalWrite(DIRB1,LOW);  digitalWrite(DIRB2,LOW);  analogWrite(PWMB,0);}while(0)

#define MOTORC_FORWARD(pwm)    do{digitalWrite(DIRC1,HIGH); digitalWrite(DIRC2,LOW);  analogWrite(PWMC,pwm);}while(0)
#define MOTORC_BACKOFF(pwm)    do{digitalWrite(DIRC1,LOW);  digitalWrite(DIRC2,HIGH); analogWrite(PWMC,pwm);}while(0)
#define MOTORC_STOP()          do{digitalWrite(DIRC1,LOW);  digitalWrite(DIRC2,LOW);  analogWrite(PWMC,0);}while(0)

#define MOTORD_FORWARD(pwm)    do{digitalWrite(DIRD1,LOW);  digitalWrite(DIRD2,HIGH); analogWrite(PWMD,pwm);}while(0)
#define MOTORD_BACKOFF(pwm)    do{digitalWrite(DIRD1,HIGH); digitalWrite(DIRD2,LOW);  analogWrite(PWMD,pwm);}while(0)
#define MOTORD_STOP()          do{digitalWrite(DIRD1,LOW);  digitalWrite(DIRD2,LOW);  analogWrite(PWMD,0);}while(0)

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

int scaledPwm(float v) {
  float a = fabs(v);
  if (a < 0.001f) return 0;

  int pwm = (int)(a * Motor_PWM + 0.5f);
  if (pwm > 0 && pwm < MIN_MOTION_PWM) pwm = MIN_MOTION_PWM;
  if (pwm > 255) pwm = 255;
  return pwm;
}

void motorA_signed(float v) { // RF
  int pwm = scaledPwm(v);
  if (pwm == 0) MOTORA_STOP();
  else if (v > 0) MOTORA_FORWARD(pwm);
  else MOTORA_BACKOFF(pwm);
}

void motorB_signed(float v) { // LF
  int pwm = scaledPwm(v);
  if (pwm == 0) MOTORB_STOP();
  else if (v > 0) MOTORB_FORWARD(pwm);
  else MOTORB_BACKOFF(pwm);
}

void motorC_signed(float v) { // RR
  int pwm = scaledPwm(v);
  if (pwm == 0) MOTORC_STOP();
  else if (v > 0) MOTORC_FORWARD(pwm);
  else MOTORC_BACKOFF(pwm);
}

void motorD_signed(float v) { // LR
  int pwm = scaledPwm(v);
  if (pwm == 0) MOTORD_STOP();
  else if (v > 0) MOTORD_FORWARD(pwm);
  else MOTORD_BACKOFF(pwm);
}

float slewWheel(float current, float target) {
  float diff = target - current;
  if (fabs(diff) < 0.0005f) return target;

  float step = (fabs(target) < fabs(current)) ? DECEL_STEP : ACCEL_STEP;
  if (fabs(diff) <= step) return target;
  return current + (diff > 0.0f ? step : -step);
}

void setWheelTargets(float lf, float rf, float lr, float rr) {
  targetLF = clampf(lf, -1.0f, 1.0f);
  targetRF = clampf(rf, -1.0f, 1.0f);
  targetLR = clampf(lr, -1.0f, 1.0f);
  targetRR = clampf(rr, -1.0f, 1.0f);
}

void applyCurrentWheels() {
  // actual motor mapping: A=RF, B=LF, C=RR, D=LR
  motorB_signed(currentLF);
  motorA_signed(currentRF);
  motorD_signed(currentLR);
  motorC_signed(currentRR);
}

void updateMotorRamp() {
  unsigned long now = millis();
  if (now - lastRampMs < RAMP_PERIOD_MS) return;
  lastRampMs = now;

  currentLF = slewWheel(currentLF, targetLF);
  currentRF = slewWheel(currentRF, targetRF);
  currentLR = slewWheel(currentLR, targetLR);
  currentRR = slewWheel(currentRR, targetRR);

  applyCurrentWheels();
}

void stopMotors() {
  targetLF = targetRF = targetLR = targetRR = 0.0f;
  currentLF = currentRF = currentLR = currentRR = 0.0f;

  MOTORA_STOP();
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_STOP();
}

// ===== Legacy key motions =====
void moveForward() {
  motorA_signed(1.0f);
  motorB_signed(1.0f);
  motorC_signed(1.0f);
  motorD_signed(1.0f);
}

void moveBackward() {
  motorA_signed(-1.0f);
  motorB_signed(-1.0f);
  motorC_signed(-1.0f);
  motorD_signed(-1.0f);
}

void turnLeft() {
  motorA_signed(-1.0f);
  motorB_signed(1.0f);
  motorC_signed(-1.0f);
  motorD_signed(1.0f);
}

void turnRight() {
  motorA_signed(1.0f);
  motorB_signed(-1.0f);
  motorC_signed(1.0f);
  motorD_signed(-1.0f);
}

void strafeLeft() {
  motorA_signed(-1.0f);
  motorB_signed(1.0f);
  motorC_signed(1.0f);
  motorD_signed(-1.0f);
}

void strafeRight() {
  motorA_signed(1.0f);
  motorB_signed(-1.0f);
  motorC_signed(-1.0f);
  motorD_signed(1.0f);
}

void diagFrontLeft() {
  motorA_signed(0.0f);
  motorB_signed(1.0f);
  motorC_signed(1.0f);
  motorD_signed(0.0f);
}

void diagFrontRight() {
  motorA_signed(1.0f);
  motorB_signed(0.0f);
  motorC_signed(0.0f);
  motorD_signed(1.0f);
}

void diagBackLeft() {
  motorA_signed(-1.0f);
  motorB_signed(0.0f);
  motorC_signed(0.0f);
  motorD_signed(-1.0f);
}

void diagBackRight() {
  motorA_signed(0.0f);
  motorB_signed(-1.0f);
  motorC_signed(-1.0f);
  motorD_signed(0.0f);
}

// ===== Continuous mecanum velocity command =====
// vx: forward +, vy: left +, wz: CCW/left-turn +
// wheel order for kinematics: LF RF LR RR
void driveVelocity(float vx, float vy, float wz) {
  vx = clampf(vx, -1.0f, 1.0f);
  vy = clampf(vy, -1.0f, 1.0f);
  wz = clampf(wz, -1.0f, 1.0f);

  float lf = vx + vy + wz;
  float rf = vx - vy - wz;
  float lr = vx - vy + wz;
  float rr = vx + vy - wz;

  float maxAbs = fabs(lf);
  if (fabs(rf) > maxAbs) maxAbs = fabs(rf);
  if (fabs(lr) > maxAbs) maxAbs = fabs(lr);
  if (fabs(rr) > maxAbs) maxAbs = fabs(rr);

  if (maxAbs > 1.0f) {
    lf /= maxAbs;
    rf /= maxAbs;
    lr /= maxAbs;
    rr /= maxAbs;
  }

  // actual motor mapping is applied inside updateMotorRamp().
  // Here we only set target wheel commands for smooth acceleration/deceleration.
  setWheelTargets(lf, rf, lr, rr);

  lastCmdMs = millis();
}

void publishEncoder() {
  long rawA = encA.read();     // actual RF
  long rawB = -encB.read();    // actual LF
  long rawC = encC.read();     // actual RR
  long rawD = -encD.read();    // actual LR

  long lf = rawB;
  long rf = rawA;
  long lr = rawD;
  long rr = rawC;

  Serial.print("ENC ");
  Serial.print(lf);
  Serial.print(" ");
  Serial.print(rf);
  Serial.print(" ");
  Serial.print(lr);
  Serial.print(" ");
  Serial.println(rr);
}

void handleKeyCommand(char cmd) {
  if (cmd == '\n' || cmd == '\r') return;

  switch (cmd) {
    case 'w': case 'W': moveForward();  lastCmdMs = millis(); break;
    case 's': case 'S': moveBackward(); lastCmdMs = millis(); break;
    case 'a': case 'A': turnLeft();     lastCmdMs = millis(); break;
    case 'd': case 'D': turnRight();    lastCmdMs = millis(); break;
    case 'q': case 'Q': strafeLeft();   lastCmdMs = millis(); break;
    case 'e': case 'E': strafeRight();  lastCmdMs = millis(); break;
    case 'r': case 'R': diagFrontLeft();  lastCmdMs = millis(); break;
    case 't': case 'T': diagFrontRight(); lastCmdMs = millis(); break;
    case 'f': case 'F': diagBackLeft();   lastCmdMs = millis(); break;
    case 'g': case 'G': diagBackRight();  lastCmdMs = millis(); break;

    case 'x': case 'X':
      stopMotors();
      lastCmdMs = millis();
      break;

    case 'l': case 'L':
      Motor_PWM += 10;
      if (Motor_PWM > 255) Motor_PWM = 255;
      Serial.print("PWM "); Serial.println(Motor_PWM);
      break;

    case 'm': case 'M':
      Motor_PWM -= 10;
      if (Motor_PWM < 0) Motor_PWM = 0;
      Serial.print("PWM "); Serial.println(Motor_PWM);
      break;
  }
}

void handleLine(char *line) {
  char *cmd = strtok(line, " ,\t");
  if (cmd == NULL) return;

  if (cmd[0] == 'V' || cmd[0] == 'v') {
    char *sx = strtok(NULL, " ,\t");
    char *sy = strtok(NULL, " ,\t");
    char *sw = strtok(NULL, " ,\t");

    if (sx == NULL || sy == NULL || sw == NULL) {
      stopMotors();
      return;
    }

    float vx = atof(sx);
    float vy = atof(sy);
    float wz = atof(sw);

    driveVelocity(vx, vy, wz);
    return;
  }

  if (cmd[0] == 'P' || cmd[0] == 'p') {
    char *spwm = strtok(NULL, " ,\t");
    if (spwm != NULL) {
      Motor_PWM = atoi(spwm);
      if (Motor_PWM < 0) Motor_PWM = 0;
      if (Motor_PWM > 255) Motor_PWM = 255;
      Serial.print("PWM "); Serial.println(Motor_PWM);
    }
    return;
  }

  if (cmd[1] == '\0') {
    handleKeyCommand(cmd[0]);
  }
}

void readSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();

    if (lineMode) {
      if (c == '\n' || c == '\r') {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
        lineMode = false;
      } else if (lineLen < sizeof(lineBuf) - 1) {
        lineBuf[lineLen++] = c;
      } else {
        lineLen = 0;
        lineMode = false;
        stopMotors();
      }
      continue;
    }

    if (c == 'V' || c == 'v' || c == 'P' || c == 'p') {
      lineMode = true;
      lineLen = 0;
      lineBuf[lineLen++] = c;
      continue;
    }

    handleKeyCommand(c);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(PWMA, OUTPUT);
  pinMode(DIRA1, OUTPUT);
  pinMode(DIRA2, OUTPUT);

  pinMode(PWMB, OUTPUT);
  pinMode(DIRB1, OUTPUT);
  pinMode(DIRB2, OUTPUT);

  pinMode(PWMC, OUTPUT);
  pinMode(DIRC1, OUTPUT);
  pinMode(DIRC2, OUTPUT);

  pinMode(PWMD, OUTPUT);
  pinMode(DIRD1, OUTPUT);
  pinMode(DIRD2, OUTPUT);

  stopMotors();

  lastCmdMs = millis();
  lastEncMs = millis();

  Serial.println("READY VELOCITY_WITH_ENCODER");
  Serial.println("CMD: V vx vy wz  where vx,vy,wz in [-1,1]");
  Serial.println("KEYS: w/s/a/d/q/e/r/t/f/g/x/l/m");
  Serial.println("FORMAT: ENC LF RF LR RR");
}

void loop() {
  unsigned long now = millis();

  readSerialCommands();

  updateMotorRamp();

  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    stopMotors();
  }

  if (now - lastEncMs >= ENC_PERIOD_MS) {
    lastEncMs = now;
    publishEncoder();
  }
}
