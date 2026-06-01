#!/usr/bin/env python3

import sys
import time
import termios
import tty
import select

try:
    import serial
except ImportError:
    print("pyserial 없음: sudo apt install python3-serial")
    sys.exit(1)


PORT = "/dev/arduino"
BAUD = 115200


HELP = """
[Keyboard Serial Drive]
  w : forward
  s : backward
  a : turn left
  d : turn right
  q : strafe left
  e : strafe right
  r : diagonal front-left
  t : diagonal front-right
  f : diagonal back-left
  g : diagonal back-right
  x : stop
  l : PWM +10
  m : PWM -10
  space : stop
  Ctrl+C : quit

먼저 l을 5~10번 눌러 PWM을 올린 뒤 w를 눌러보십시오.
"""


def get_key(timeout=0.05):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return None


def main():
    port = PORT
    if len(sys.argv) >= 2:
        port = sys.argv[1]

    print(f"open serial: {port} @ {BAUD}")
    ser = serial.Serial(port, BAUD, timeout=0.05)
    time.sleep(2.0)

    print(HELP)

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            key = get_key()

            if key is None:
                continue

            if key == "\x03":
                break

            if key == " ":
                key = "x"

            allowed = set("wasdqertfgxlmWASDQERTFGXLM")

            if key in allowed:
                ser.write(key.encode("ascii"))
                ser.flush()
                print(f"sent: {key}")

            # Arduino가 encoder를 계속 뿌리므로, 수신 버퍼 일부 출력
            while ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"arduino: {line}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.write(b"x")
            ser.flush()
            time.sleep(0.1)
            ser.close()
        except Exception:
            pass

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nstopped")


if __name__ == "__main__":
    main()
