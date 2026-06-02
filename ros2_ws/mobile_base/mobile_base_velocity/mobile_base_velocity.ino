#include <Encoder.h>
#include <math.h>
#include <string.h>

// ============================================================
// Ze-Ri Mobile Base Arduino Firmware
//
// Protocol:
//   V vx vy wz\n     vx, vy, wz in [-1.0, 1.0]
//   P pwm\n          set max PWM
//   X\n              smooth stop
//   Z\n              immediate stop
//
// IMPORTANT:
//   Acceleration command is NOT accepted.
//   There is NO "A accel decel" serial command.
//   Ramp values are fixed inside this firmware.
//
// Legacy lowercase keys for manual test:
//   w/s : forward/backward
//   a/d : turn left/right
//   q/e : strafe left/right
//   r/t/f/g : diagonal
//   x : smooth stop
//   z : immediate stop
//   l/m : PWM +10 / -10
//
// Encoder output:
//   ENC LF RF LR RR
// ============================================================


// ============================================================
// Motor pins
// ============================================================

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


// ============================================================
// Encoder pins
//
// Actual motor mapping:
//   A = RF
//   B = LF, inverted
//   C = RR
//   D = LR, inverted
// ============================================================

Encoder encA(18, 31);   // RF
Encoder encB(19, 38);   // LF, inverted
Encoder encC(3, 49);    // RR
Encoder encD(2, A1);    // LR, inverted


// ============================================================
// Speed / timing
// ============================================================

int Motor_PWM = 60;

const int MIN_MOTION_PWM = 18;

const unsigned long ENC_PERIOD_MS = 100;
const unsigned long CMD_TIMEOUT_MS = 800;
const unsigned long RAMP_PERIOD_MS = 20;

unsigned long lastEncMs = 0;
unsigned long lastCmdMs = 0;
unsigned long lastRampMs = 0;


// ============================================================
// Fixed internal ramp
//
// These values are NOT configurable through serial.
// Arduino must NOT receive acceleration/deceleration values.
// ============================================================

const float RAMP_UP_STEP = 0.025f;
const float RAMP_DOWN_STEP = 0.040f;


// ============================================================
// Wheel command state
//
// Wheel order:
//   LF RF LR RR
// ============================================================

float targetLF = 0.0f;
float targetRF = 0.0f;
float targetLR = 0.0f;
float targetRR = 0.0f;

float currentLF = 0.0f;
float currentRF = 0.0f;
float currentLR = 0.0f;
float currentRR = 0.0f;


// ============================================================
// Serial line buffer
// ============================================================

char lineBuf[96];
uint8_t lineLen = 0;
bool lineMode = false;


// ============================================================
// Motor low-level macros
// ============================================================

#define MOTORA_FORWARD(pwm)    do { digitalWrite(DIRA1, HIGH); digitalWrite(DIRA2, LOW);  analogWrite(PWMA, pwm); } while (0)
#define MOTORA_BACKOFF(pwm)    do { digitalWrite(DIRA1, LOW);  digitalWrite(DIRA2, HIGH); analogWrite(PWMA, pwm); } while (0)
#define MOTORA_STOP()          do { digitalWrite(DIRA1, LOW);  digitalWrite(DIRA2, LOW);  analogWrite(PWMA, 0);   } while (0)

#define MOTORB_FORWARD(pwm)    do { digitalWrite(DIRB1, LOW);  digitalWrite(DIRB2, HIGH); analogWrite(PWMB, pwm); } while (0)
#define MOTORB_BACKOFF(pwm)    do { digitalWrite(DIRB1, HIGH); digitalWrite(DIRB2, LOW);  analogWrite(PWMB, pwm); } while (0)
#define MOTORB_STOP()          do { digitalWrite(DIRB1, LOW);  digitalWrite(DIRB2, LOW);  analogWrite(PWMB, 0);   } while (0)

#define MOTORC_FORWARD(pwm)    do { digitalWrite(DIRC1, HIGH); digitalWrite(DIRC2, LOW);  analogWrite(PWMC, pwm); } while (0)
#define MOTORC_BACKOFF(pwm)    do { digitalWrite(DIRC1, LOW);  digitalWrite(DIRC2, HIGH); analogWrite(PWMC, pwm); } while (0)
#define MOTORC_STOP()          do { digitalWrite(DIRC1, LOW);  digitalWrite(DIRC2, LOW);  analogWrite(PWMC, 0);   } while (0)

#define MOTORD_FORWARD(pwm)    do { digitalWrite(DIRD1, LOW);  digitalWrite(DIRD2, HIGH); analogWrite(PWMD, pwm); } while (0)
#define MOTORD_BACKOFF(pwm)    do { digitalWrite(DIRD1, HIGH); digitalWrite(DIRD2, LOW);  analogWrite(PWMD, pwm); } while (0)
#define MOTORD_STOP()          do { digitalWrite(DIRD1, LOW);  digitalWrite(DIRD2, LOW);  analogWrite(PWMD, 0);   } while (0)


// ============================================================
// Utility
// ============================================================

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

bool oppositeSign(float a, float b, float eps) {
  return fabs(a) > eps && fabs(b) > eps && (a * b) < 0.0f;
}

int scaledPwm(float v) {
  float a = fabs(v);

  if (a < 0.001f) {
    return 0;
  }

  int pwm = (int)(a * Motor_PWM + 0.5f);

  if (pwm > 0 && pwm < MIN_MOTION_PWM) {
    pwm = MIN_MOTION_PWM;
  }

  if (pwm > 255) {
    pwm = 255;
  }

  return pwm;
}


// ============================================================
// Motor signed output
// ============================================================

void motorA_signed(float v) {
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORA_STOP();
  } else if (v > 0.0f) {
    MOTORA_FORWARD(pwm);
  } else {
    MOTORA_BACKOFF(pwm);
  }
}

void motorB_signed(float v) {
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORB_STOP();
  } else if (v > 0.0f) {
    MOTORB_FORWARD(pwm);
  } else {
    MOTORB_BACKOFF(pwm);
  }
}

void motorC_signed(float v) {
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORC_STOP();
  } else if (v > 0.0f) {
    MOTORC_FORWARD(pwm);
  } else {
    MOTORC_BACKOFF(pwm);
  }
}

void motorD_signed(float v) {
  int pwm = scaledPwm(v);

  if (pwm == 0) {
    MOTORD_STOP();
  } else if (v > 0.0f) {
    MOTORD_FORWARD(pwm);
  } else {
    MOTORD_BACKOFF(pwm);
  }
}


// ============================================================
// Wheel ramp
// ============================================================

float slewWheel(float current, float target) {
  target = clampf(target, -1.0f, 1.0f);

  // Avoid direct reverse. Pass through zero first.
  if (oppositeSign(current, target, 0.02f)) {
    target = 0.0f;
  }

  float diff = target - current;

  if (fabs(diff) < 0.0005f) {
    return target;
  }

  float step;

  if (fabs(target) < fabs(current)) {
    step = RAMP_DOWN_STEP;
  } else {
    step = RAMP_UP_STEP;
  }

  if (fabs(diff) <= step) {
    return target;
  }

  if (diff > 0.0f) {
    return current + step;
  }

  return current - step;
}

void setWheelTargets(float lf, float rf, float lr, float rr) {
  targetLF = clampf(lf, -1.0f, 1.0f);
  targetRF = clampf(rf, -1.0f, 1.0f);
  targetLR = clampf(lr, -1.0f, 1.0f);
  targetRR = clampf(rr, -1.0f, 1.0f);
}

void applyCurrentWheels() {
  // Actual motor mapping:
  //   A = RF
  //   B = LF
  //   C = RR
  //   D = LR
  motorB_signed(currentLF);
  motorA_signed(currentRF);
  motorD_signed(currentLR);
  motorC_signed(currentRR);
}

void updateMotorRamp() {
  unsigned long now = millis();

  if (now - lastRampMs < RAMP_PERIOD_MS) {
    return;
  }

  lastRampMs = now;

  currentLF = slewWheel(currentLF, targetLF);
  currentRF = slewWheel(currentRF, targetRF);
  currentLR = slewWheel(currentLR, targetLR);
  currentRR = slewWheel(currentRR, targetRR);

  applyCurrentWheels();
}

void immediateStopMotors() {
  targetLF = 0.0f;
  targetRF = 0.0f;
  targetLR = 0.0f;
  targetRR = 0.0f;

  currentLF = 0.0f;
  currentRF = 0.0f;
  currentLR = 0.0f;
  currentRR = 0.0f;

  MOTORA_STOP();
  MOTORB_STOP();
  MOTORC_STOP();
  MOTORD_STOP();
}

void smoothStopMotors() {
  setWheelTargets(0.0f, 0.0f, 0.0f, 0.0f);
}


// ============================================================
// Mecanum velocity command
//
// vx: forward +
// vy: left +
// wz: CCW / left turn +
// ============================================================

void driveVelocity(float vx, float vy, float wz) {
  vx = clampf(vx, -1.0f, 1.0f);
  vy = clampf(vy, -1.0f, 1.0f);
  wz = clampf(wz, -1.0f, 1.0f);

  // Wheel order:
  //   LF RF LR RR
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

  setWheelTargets(lf, rf, lr, rr);
  lastCmdMs = millis();
}


// ============================================================
// Legacy lowercase key command
//
// Uppercase A is intentionally NOT used as turn command.
// This prevents accidental old "A accel decel" input from moving robot.
// ============================================================

bool isLegacyKey(char cmd) {
  return (
    cmd == 'w' ||
    cmd == 's' ||
    cmd == 'a' ||
    cmd == 'd' ||
    cmd == 'q' ||
    cmd == 'e' ||
    cmd == 'r' ||
    cmd == 't' ||
    cmd == 'f' ||
    cmd == 'g' ||
    cmd == 'x' ||
    cmd == 'z' ||
    cmd == 'l' ||
    cmd == 'm'
  );
}

void handleKeyCommand(char cmd) {
  if (cmd == '\n' || cmd == '\r' || cmd == ' ' || cmd == '\t') {
    return;
  }

  switch (cmd) {
    case 'w':
      driveVelocity(1.0f, 0.0f, 0.0f);
      break;

    case 's':
      driveVelocity(-1.0f, 0.0f, 0.0f);
      break;

    case 'a':
      driveVelocity(0.0f, 0.0f, 1.0f);
      break;

    case 'd':
      driveVelocity(0.0f, 0.0f, -1.0f);
      break;

    case 'q':
      driveVelocity(0.0f, 1.0f, 0.0f);
      break;

    case 'e':
      driveVelocity(0.0f, -1.0f, 0.0f);
      break;

    case 'r':
      driveVelocity(0.7f, 0.7f, 0.0f);
      break;

    case 't':
      driveVelocity(0.7f, -0.7f, 0.0f);
      break;

    case 'f':
      driveVelocity(-0.7f, 0.7f, 0.0f);
      break;

    case 'g':
      driveVelocity(-0.7f, -0.7f, 0.0f);
      break;

    case 'x':
      smoothStopMotors();
      lastCmdMs = millis();
      break;

    case 'z':
      immediateStopMotors();
      lastCmdMs = millis();
      break;

    case 'l':
      Motor_PWM += 10;
      if (Motor_PWM > 255) Motor_PWM = 255;
      Serial.print("PWM ");
      Serial.println(Motor_PWM);
      break;

    case 'm':
      Motor_PWM -= 10;
      if (Motor_PWM < 0) Motor_PWM = 0;
      Serial.print("PWM ");
      Serial.println(Motor_PWM);
      break;

    default:
      break;
  }
}


// ============================================================
// Encoder publish
//
// Output order must match base node:
//   ENC LF RF LR RR
// ============================================================

void publishEncoder() {
  long rawA = encA.read();     // RF
  long rawB = -encB.read();    // LF inverted
  long rawC = encC.read();     // RR
  long rawD = -encD.read();    // LR inverted

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


// ============================================================
// Line protocol
// ============================================================

void handleLine(char *line) {
  char *cmd = strtok(line, " ,\t");

  if (cmd == NULL) {
    return;
  }

  // V vx vy wz
  if (cmd[0] == 'V' || cmd[0] == 'v') {
    char *sx = strtok(NULL, " ,\t");
    char *sy = strtok(NULL, " ,\t");
    char *sw = strtok(NULL, " ,\t");

    if (sx == NULL || sy == NULL || sw == NULL) {
      smoothStopMotors();
      return;
    }

    float vx = atof(sx);
    float vy = atof(sy);
    float wz = atof(sw);

    driveVelocity(vx, vy, wz);
    return;
  }

  // P pwm
  if (cmd[0] == 'P' || cmd[0] == 'p') {
    char *spwm = strtok(NULL, " ,\t");

    if (spwm != NULL) {
      Motor_PWM = atoi(spwm);

      if (Motor_PWM < 0) Motor_PWM = 0;
      if (Motor_PWM > 255) Motor_PWM = 255;

      Serial.print("PWM ");
      Serial.println(Motor_PWM);
    }

    return;
  }

  // X smooth stop
  if (cmd[0] == 'X' || cmd[0] == 'x') {
    smoothStopMotors();
    lastCmdMs = millis();
    return;
  }

  // Z immediate stop
  if (cmd[0] == 'Z' || cmd[0] == 'z') {
    immediateStopMotors();
    lastCmdMs = millis();
    return;
  }

  // No acceleration command is accepted.
  // Old "A accel decel" lines are intentionally ignored.
  if (cmd[0] == 'A') {
    return;
  }

  // Single lowercase key fallback
  if (cmd[1] == '\0' && isLegacyKey(cmd[0])) {
    handleKeyCommand(cmd[0]);
    return;
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
        immediateStopMotors();
      }

      continue;
    }

    // Line-based commands.
    //
    // V/P are valid line commands.
    // A is consumed as a line only to safely ignore old acceleration input.
    // It does NOT configure acceleration.
    if (c == 'V' || c == 'v' ||
        c == 'P' || c == 'p' ||
        c == 'A') {
      lineMode = true;
      lineLen = 0;
      lineBuf[lineLen++] = c;
      continue;
    }

    // Direct immediate commands
    if (c == 'X') {
      smoothStopMotors();
      lastCmdMs = millis();
      continue;
    }

    if (c == 'Z') {
      immediateStopMotors();
      lastCmdMs = millis();
      continue;
    }

    // Single lowercase key commands
    if (isLegacyKey(c)) {
      handleKeyCommand(c);
      continue;
    }

    // Ignore everything else
  }
}


// ============================================================
// Setup / loop
// ============================================================

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

  immediateStopMotors();

  lastCmdMs = millis();
  lastEncMs = millis();
  lastRampMs = millis();

  Serial.println("READY ZE_RI_BASE_VELOCITY_FIXED_RAMP");
  Serial.println("CMD: V vx vy wz  where vx,vy,wz in [-1,1]");
  Serial.println("CMD: P pwm");
  Serial.println("CMD: X smooth stop");
  Serial.println("CMD: Z immediate stop");
  Serial.println("NO CMD: A accel decel");
  Serial.println("KEYS lowercase only: w/s/a/d/q/e/r/t/f/g/x/z/l/m");
  Serial.println("FORMAT: ENC LF RF LR RR");
}

void loop() {
  unsigned long now = millis();

  readSerialCommands();
  updateMotorRamp();

  // If command stream stops, stop safely.
  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    immediateStopMotors();
    lastCmdMs = now;
  }

  if (now - lastEncMs >= ENC_PERIOD_MS) {
    lastEncMs = now;
    publishEncoder();
  }
}
