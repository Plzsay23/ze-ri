#!/usr/bin/env python3

import sys
import time
import termios
import tty
import select

import serial


PORT = "/dev/arduino"
BAUD = 115200

HELP = """
[Keyboard Hold Drive]

키를 누르면 해당 명령을 2초 동안 반복 전송합니다.

  w : forward
  s : backward
  a : turn left
  d : turn right
  q : strafe left
  e : strafe right
  x : stop
  l : PWM +10
  m : PWM -10
  Ctrl+C : quit

추천 테스트:
  1) l 을 10번 눌러 PWM 올림
  2) w 누름
  3) x 정지
"""


def get_key():
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        return sys.stdin.read(1)
    return None


def drain_serial(ser):
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(f"arduino: {line}")


def main():
    port = PORT
    if len(sys.argv) >= 2:
        port = sys.argv[1]

    print(f"open serial: {port} @ {BAUD}")
    ser = serial.Serial(port, BAUD, timeout=0.02)
    time.sleep(2.0)

    print(HELP)

    old = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            drain_serial(ser)

            key = get_key()
            if key is None:
                continue

            if key == "\x03":
                break

            if key == " ":
                key = "x"

            if key not in "wasdqexlmWASDQEXLM":
                continue

            print(f"pressed: {key}")

            if key.lower() in ["l", "m", "x"]:
                ser.write(key.encode("ascii"))
                ser.flush()
                time.sleep(0.1)
                drain_serial(ser)
                continue

            # 이동 명령은 2초 동안 20Hz로 반복 전송
            start = time.time()
            while time.time() - start < 2.0:
                ser.write(key.encode("ascii"))
                ser.flush()
                drain_serial(ser)
                time.sleep(0.05)

            ser.write(b"x")
            ser.flush()
            print("sent: x")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"x")
            ser.flush()
            ser.close()
        except Exception:
            pass

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("stopped")


if __name__ == "__main__":
    main()
