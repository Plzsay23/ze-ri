.PHONY: smoke

smoke:
	python3 -m py_compile \
		ros2_ws/src/nb_voice_stt/nb_voice_stt/*.py \
		ros2_ws/src/zeri_base/zeri_base/*.py \
		ros2_ws/src/zeri_bringup/zeri_bringup/*.py \
		ros2_ws/src/zeri_camera/zeri_camera/*.py \
		ros2_ws/src/zeri_lidar/zeri_lidar/*.py \
		ros2_ws/src/zeri_voice/zeri_voice/*.py \
		src/lerobot/vlm_agent/*.py \
		src/lerobot/configs/train.py \
		src/lerobot/scripts/lerobot_train.py \
		src/lerobot/scripts/lerobot_rollout.py \
		src/lerobot/policies/factory.py \
		src/lerobot/policies/__init__.py \
		src/lerobot/policies/pretrained.py \
		src/lerobot/policies/pi0/modeling_pi0.py \
		src/lerobot/policies/pi05/modeling_pi05.py \
		src/lerobot/processor/newline_task_processor.py \
		src/lerobot/data_processing/__init__.py \
		src/lerobot/async_inference/helpers.py \
		src/lerobot/async_inference/constants.py \
		src/lerobot/utils/import_utils.py
	python3 -c 'from pathlib import Path; import tomllib; import xml.etree.ElementTree as ET; tomllib.load(open("pyproject.toml", "rb")); [ET.parse(p) for p in sorted(Path("ros2_ws/src").glob("*/package.xml"))]; print("metadata ok")'
	bash -n source_zeri.sh source_zeri_vlm.sh scripts/*.sh
	git diff --check
