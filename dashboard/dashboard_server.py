#!/usr/bin/env python3
# zeri_dashboard_server.py

import argparse
import asyncio
import base64
import json
import math
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from std_msgs.msg import String


INDEX_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Ze-Ri ROS2 Autonomy / VLA Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <style>
    :root {
      --bg0: #020617;
      --bg1: #07111f;
      --panel: rgba(15, 23, 42, 0.92);
      --panel2: rgba(15, 35, 68, 0.82);
      --panel3: rgba(8, 16, 32, 0.82);
      --line: rgba(96, 165, 250, 0.34);
      --line2: rgba(148, 163, 184, 0.18);
      --text: #e5f0ff;
      --muted: #93a4bd;
      --soft: #bfdbfe;
      --blue: #38bdf8;
      --green: #22c55e;
      --yellow: #f59e0b;
      --red: #ef4444;
      --shadow: 0 18px 40px rgba(0, 0, 0, 0.34);
    }

    * { box-sizing: border-box; }

    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
    }

    body {
      background:
        radial-gradient(circle at 18% 10%, rgba(56, 189, 248, 0.18), transparent 28%),
        radial-gradient(circle at 82% 18%, rgba(37, 99, 235, 0.15), transparent 30%),
        linear-gradient(135deg, #020617 0%, #07111f 50%, #020617 100%);
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      overflow: hidden;
    }

    .topbar {
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: rgba(2, 6, 23, 0.88);
      border-bottom: 1px solid rgba(96, 165, 250, 0.28);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.25);
      backdrop-filter: blur(10px);
    }

    .brand {
      min-width: 0;
      display: flex;
      flex-direction: column;
      justify-content: center;
      line-height: 1.05;
    }

    .eyebrow {
      color: var(--blue);
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 1.8px;
      text-transform: uppercase;
    }

    .title {
      margin-top: 5px;
      font-size: 20px;
      font-weight: 950;
      letter-spacing: 0.2px;
      color: #f8fbff;
      white-space: nowrap;
    }

    .top-status {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      min-width: 0;
      flex-wrap: nowrap;
    }

    .status-chip {
      height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: rgba(15, 23, 42, 0.78);
      color: #cbd5e1;
      font-size: 11px;
      font-weight: 900;
      display: flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    }

    .status-chip.ok {
      color: #d1fae5;
      border-color: rgba(34, 197, 94, 0.46);
      background: rgba(6, 78, 59, 0.42);
    }

    .status-chip.warn {
      color: #fde68a;
      border-color: rgba(245, 158, 11, 0.48);
      background: rgba(120, 53, 15, 0.42);
    }

    .status-chip.bad {
      color: #fecaca;
      border-color: rgba(239, 68, 68, 0.45);
      background: rgba(127, 29, 29, 0.42);
    }

    .dot, .chip-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--red);
      display: inline-block;
      box-shadow: 0 0 10px rgba(239, 68, 68, 0.7);
    }

    .dot.ok, .status-chip.ok .chip-dot {
      background: var(--green);
      box-shadow: 0 0 12px rgba(34, 197, 94, 0.82);
    }

    .status-chip.warn .chip-dot {
      background: var(--yellow);
      box-shadow: 0 0 12px rgba(245, 158, 11, 0.7);
    }

    .status-chip.bad .chip-dot {
      background: var(--red);
      box-shadow: 0 0 12px rgba(239, 68, 68, 0.75);
    }

    .clock {
      color: #94a3b8;
      font-size: 11px;
      font-weight: 800;
      min-width: 132px;
      text-align: right;
    }

    .mission-bar {
      height: 50px;
      margin: 10px 10px 0;
      padding: 0 14px;
      border: 1px solid rgba(96, 165, 250, 0.28);
      border-radius: 14px;
      background: linear-gradient(90deg, rgba(15, 23, 42, 0.88), rgba(15, 35, 68, 0.72));
      box-shadow: var(--shadow);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }

    .mission-left {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }

    .mission-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 950;
      letter-spacing: 1.3px;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .mission-state {
      padding: 5px 12px;
      border-radius: 999px;
      color: #e0f2fe;
      background: rgba(14, 165, 233, 0.22);
      border: 1px solid rgba(56, 189, 248, 0.46);
      font-size: 14px;
      font-weight: 950;
      letter-spacing: 0.35px;
      white-space: nowrap;
    }

    .mission-desc {
      color: #cbd5e1;
      font-size: 12px;
      font-weight: 800;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .mission-metrics {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      flex-wrap: nowrap;
    }

    .mini-metric {
      min-width: 76px;
      padding: 5px 9px;
      border-radius: 10px;
      background: rgba(2, 6, 23, 0.35);
      border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .mini-metric span {
      display: block;
      color: var(--muted);
      font-size: 9px;
      font-weight: 900;
      letter-spacing: 0.8px;
    }

    .mini-metric strong {
      display: block;
      color: #f8fafc;
      font-size: 13px;
      font-weight: 950;
      margin-top: 1px;
    }

    .grid {
      width: 100vw;
      height: calc(100vh - 118px);
      display: grid;
      grid-template-columns: repeat(14, minmax(0, 1fr));
      grid-template-rows: minmax(0, 58fr) minmax(0, 42fr);
      grid-template-areas:
        "rgb rgb rgb rgb rgb rgb rgb depth depth depth depth depth depth depth"
        "map map map voice voice voice voice base base base arm arm arm arm";
      gap: 10px;
      padding: 10px;
      margin: 0;
    }

    .card {
      position: relative;
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(15, 35, 68, 0.84));
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      color: var(--text);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
    }

    .card::before {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      border-radius: inherit;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), transparent 22%);
      opacity: 0.7;
      z-index: 0;
    }

    .card > * { position: relative; z-index: 1; }

    .rgb-card { grid-area: rgb; }
    .depth-card { grid-area: depth; }
    .text-card { grid-area: voice; }
    .map-card { grid-area: map; }
    .base-card { grid-area: base; }
    .arm-card { grid-area: arm; }

    .card-title {
      height: 38px;
      flex: 0 0 38px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--line2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-width: 0;
      background: rgba(2, 6, 23, 0.26);
    }

    .card-title span {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      font-weight: 950;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      color: #f8fafc;
      min-width: 0;
      white-space: nowrap;
    }

    .card-title span::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 2px;
      background: var(--blue);
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.9);
    }

    .card-title small {
      font-size: 10px;
      color: var(--muted);
      font-weight: 850;
      text-align: right;
      max-width: 55%;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }

    .media-wrap {
      position: relative;
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #020617;
      overflow: hidden;
    }

    .media-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
      background: #020617;
    }

    .rgb-card .media-wrap img {
      object-fit: cover;
    }

    .pointcloud-wrap {
      position: relative;
      width: 100%;
      height: 100%;
      background: #020617;
      overflow: hidden;
      touch-action: none;
    }

    .pointcloud-wrap canvas {
      width: 100%;
      height: 100%;
      display: block;
      background:
        radial-gradient(circle at center, rgba(14, 165, 233, 0.11), transparent 34%),
        radial-gradient(circle at 50% 100%, rgba(37, 99, 235, 0.16), transparent 40%),
        #020617;
      cursor: grab;
    }

    .pointcloud-wrap canvas.dragging { cursor: grabbing; }

    .viewer-hint, .overlay-pill {
      position: absolute;
      padding: 6px 9px;
      border-radius: 999px;
      background: rgba(2, 6, 23, 0.68);
      color: #bfdbfe;
      font-size: 10px;
      font-weight: 900;
      pointer-events: none;
      white-space: nowrap;
      border: 1px solid rgba(96, 165, 250, 0.25);
      backdrop-filter: blur(8px);
    }

    .viewer-hint { left: 10px; bottom: 8px; }
    .overlay-pill { right: 10px; bottom: 8px; }

    .placeholder {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #93c5fd;
      font-size: 18px;
      font-weight: 950;
      text-align: center;
      padding: 14px;
      background: linear-gradient(135deg, rgba(2, 6, 23, 0.8), rgba(15, 23, 42, 0.55));
    }

    .text-panel, .status-big {
      flex: 1 1 auto;
      min-height: 0;
      padding: 9px;
      overflow: auto;
      background: rgba(2, 6, 23, 0.18);
    }

    .text-panel::-webkit-scrollbar,
    .status-big::-webkit-scrollbar,
    .status-line::-webkit-scrollbar { width: 8px; height: 8px; }

    .text-panel::-webkit-scrollbar-thumb,
    .status-big::-webkit-scrollbar-thumb,
    .status-line::-webkit-scrollbar-thumb {
      background: rgba(96, 165, 250, 0.28);
      border-radius: 999px;
    }

    .metric, .status-line {
      margin-bottom: 8px;
      padding: 9px 10px;
      background: rgba(15, 23, 42, 0.58);
      border: 1px solid rgba(148, 163, 184, 0.14);
      border-radius: 11px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    }

    .metric:last-child, .status-line:last-child { margin-bottom: 0; }

    .metric.primary {
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(15, 23, 42, 0.58));
      border-color: rgba(56, 189, 248, 0.28);
    }

    .metric-label, .status-line .label {
      color: #93c5fd;
      font-size: 10px;
      font-weight: 950;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      margin-bottom: 5px;
    }

    .metric-value, .status-line .value {
      color: #f8fafc;
      font-size: 13px;
      line-height: 1.28;
      font-weight: 800;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }

    .metric-value.small, .status-line .value {
      font-size: 11px;
      line-height: 1.27;
      font-weight: 760;
    }

    .status-big {
      display: grid;
      grid-template-columns: 1fr;
      align-content: start;
    }

    @media (max-width: 1280px) {
      body { overflow-y: auto; }
      .topbar {
        height: auto;
        min-height: 58px;
        align-items: flex-start;
        gap: 8px;
        flex-direction: column;
        padding: 10px 12px;
      }
      .top-status { justify-content: flex-start; flex-wrap: wrap; }
      .clock { text-align: left; min-width: 0; }
      .mission-bar {
        height: auto;
        min-height: 58px;
        flex-direction: column;
        align-items: flex-start;
        padding: 10px;
      }
      .mission-left { flex-wrap: wrap; }
      .mission-metrics { flex-wrap: wrap; justify-content: flex-start; }
      .grid {
        height: auto;
        min-height: calc(100vh - 118px);
        grid-template-columns: 1fr;
        grid-template-rows: 420px 420px 320px 360px 320px 340px;
        grid-template-areas: "rgb" "depth" "map" "voice" "base" "arm";
      }
    }
  </style>
</head>

<body>
  <header class="topbar">
    <div class="brand">
      <div class="eyebrow">ZERO-RISK / FIELD OPERATIONS</div>
      <div class="title">Ze-Ri Mission Dashboard</div>
    </div>

    <div class="top-status">
      <div id="chip-ws" class="status-chip bad"><span id="ws-dot" class="dot"></span><span id="ws-state">DISCONNECTED</span></div>
      <div id="chip-rgb" class="status-chip bad"><span class="chip-dot"></span><span>RGB</span><b>WAIT</b></div>
      <div id="chip-pc" class="status-chip bad"><span class="chip-dot"></span><span>3D</span><b>WAIT</b></div>
      <div id="chip-map" class="status-chip bad"><span class="chip-dot"></span><span>SLAM</span><b>WAIT</b></div>
      <div id="chip-base" class="status-chip bad"><span class="chip-dot"></span><span>BASE</span><b>WAIT</b></div>
      <div id="chip-vla" class="status-chip bad"><span class="chip-dot"></span><span>VLA</span><b>WAIT</b></div>
      <div id="clock" class="clock">-</div>
    </div>
  </header>

  <section class="mission-bar">
    <div class="mission-left">
      <div class="mission-label">MISSION STATE</div>
      <div id="mission-state" class="mission-state">BOOTING</div>
      <div id="mission-desc" class="mission-desc">ROS 데이터 수신 대기중</div>
    </div>
    <div class="mission-metrics">
      <div class="mini-metric"><span>RGB FRAMES</span><strong id="rgb-count">0</strong></div>
      <div class="mini-metric"><span>3D POINTS</span><strong id="pc-count">0</strong></div>
      <div class="mini-metric"><span>MAP FRAMES</span><strong id="map-count">0</strong></div>
      <div class="mini-metric"><span>LAST UPDATE</span><strong id="last-update">-</strong></div>
    </div>
  </section>

  <main class="grid">
    <section class="card rgb-card">
      <div class="card-title">
        <span>RGB Live Feed</span>
        <small id="rgb-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="rgb-img" alt="RGB Stream" style="display:none" />
        <div id="rgb-placeholder" class="placeholder">RGB 데이터 대기중</div>
        <div id="rgb-overlay" class="overlay-pill">camera/color/image_raw</div>
      </div>
    </section>

    <section class="card depth-card">
      <div class="card-title">
        <span>3D PointCloud View</span>
        <small id="depth-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <div class="pointcloud-wrap">
          <canvas id="pointcloud-canvas"></canvas>
          <div id="pointcloud-placeholder" class="placeholder">PointCloud 데이터 대기중</div>
          <div class="viewer-hint">좌클릭 회전 · 우클릭 이동 · 휠 줌</div>
          <div id="pc-overlay" class="overlay-pill">0 points</div>
        </div>
      </div>
    </section>

    <section class="card map-card">
      <div class="card-title">
        <span>2D SLAM Map</span>
        <small id="map-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="map-img" alt="2D Map" style="display:none" />
        <div id="map-placeholder" class="placeholder">/map 데이터 대기중</div>
      </div>
    </section>

    <section class="card text-card">
      <div class="card-title">
        <span>Voice / VLM</span>
        <small id="vlm-ts">대기중</small>
      </div>
      <div class="text-panel">
        <div class="metric primary">
          <div class="metric-label">STT 입력</div>
          <div id="stt-text" class="metric-value">대기중</div>
        </div>
        <div class="metric primary">
          <div class="metric-label">TTS 출력</div>
          <div id="robot-speech" class="metric-value">대기중</div>
        </div>
        <div class="metric">
          <div class="metric-label">VLM 상태</div>
          <div id="vlm-status" class="metric-value small">대기중</div>
        </div>
        <div class="metric">
          <div class="metric-label">VLM 판단값</div>
          <div id="vlm-output" class="metric-value small">대기중</div>
        </div>
        <div class="metric">
          <div class="metric-label">판단 근거</div>
          <div id="vlm-reason" class="metric-value small">대기중</div>
        </div>
      </div>
    </section>

    <section class="card base-card">
      <div class="card-title">
        <span>Autonomy / Safety</span>
        <small id="base-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">Person Follow</div>
          <div id="person-follow-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">LiDAR + Depth Safety Guard</div>
          <div id="safety-guard-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">cmd_vel_raw → cmd_vel</div>
          <div id="cmd-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">Scan / Odom</div>
          <div id="nav-state" class="value">데이터 대기중</div>
        </div>
      </div>
    </section>

    <section class="card arm-card">
      <div class="card-title">
        <span>VLA / Robot Arms</span>
        <small id="arm-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">VLA Router</div>
          <div id="vla-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">Left Arm</div>
          <div id="vla-left-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">Right Arm</div>
          <div id="vla-right-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line">
          <div class="label">Last Task Request / Legacy Arm Status</div>
          <div id="arm-status" class="value">데이터 대기중</div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);

    function setImg(imgId, placeholderId, dataUrl) {
      const img = $(imgId);
      const ph = $(placeholderId);

      if (dataUrl) {
        img.src = dataUrl;
        img.style.display = "block";
        ph.style.display = "none";
      } else {
        img.style.display = "none";
        ph.style.display = "block";
      }
    }

    function fmtTs(ts, count) {
      if (!ts) return "대기중";
      const d = new Date(ts * 1000);
      const t = d.toLocaleTimeString();
      if (count !== undefined && count !== null) {
        return `${t} | ${count}`;
      }
      return t;
    }

    function safeText(v, fallback = "대기중") {
      if (v === null || v === undefined || v === "") return fallback;
      if (typeof v === "object") return JSON.stringify(v, null, 2);
      return String(v);
    }


    function isFresh(ts, maxAgeSec) {
      if (!ts) return false;
      return ((Date.now() / 1000.0) - ts) <= maxAgeSec;
    }

    function shortAge(ts) {
      if (!ts) return "WAIT";
      const age = Math.max(0, (Date.now() / 1000.0) - ts);
      if (age < 1.0) return "LIVE";
      if (age < 60.0) return `${age.toFixed(0)}s`;
      return `${Math.floor(age / 60)}m`;
    }

    function setChip(id, state, detail) {
      const el = $(id);
      if (!el) return;
      el.classList.remove("ok", "warn", "bad");
      el.classList.add(state);
      const b = el.querySelector("b");
      if (b) b.textContent = detail;
    }

    function hasUsefulText(v) {
      const t = safeText(v, "").trim();
      return t && t !== "데이터 대기중" && t !== "대기중" && t !== "-";
    }

    function inferMissionState(data) {
      const ts = data.timestamps || {};
      const safety = safeText(data.safety_guard_state, "").toLowerCase();
      const person = safeText(data.person_follow_state, "").toLowerCase();
      const vla = safeText(data.vla_status, "").toLowerCase();
      const vlm = safeText(data.inference_status, "").toLowerCase();

      if (isFresh(ts.vla, 4) || /running|execute|active|busy|실행/.test(vla)) {
        return ["VLA_RUNNING", "로봇팔 정책 실행 또는 라우터 상태 수신중"];
      }
      if (/block|stop|obstacle|danger|guard|emergency|충돌|장애물|정지/.test(safety)) {
        return ["SAFETY_GUARD", "LiDAR/Depth 안전 가드가 주행 명령을 감시중"];
      }
      if (/running|infer|processing|생성|추론/.test(vlm) || isFresh(ts.vlm, 3)) {
        return ["VLM_DIALOG", "STT/VLM/TTS 대화 루프가 갱신중"];
      }
      if (isFresh(ts.person, 3) || /track|follow|approach|person|사람|추종|접근/.test(person)) {
        return ["PERSON_FOLLOW", "사람 추종 및 접근 주행 상태 수신중"];
      }
      if (isFresh(ts.cmd_out, 3) || isFresh(ts.odom, 3)) {
        return ["BASE_ACTIVE", "베이스 주행 명령 또는 오도메트리 수신중"];
      }
      if (isFresh(ts.rgb, 3) && isFresh(ts.pointcloud, 3) && isFresh(ts.map, 8)) {
        return ["AUTONOMY_READY", "RGB·3D·SLAM 데이터가 정상 수신중"];
      }
      if (isFresh(ts.rgb, 3) || isFresh(ts.pointcloud, 3)) {
        return ["SENSOR_READY", "카메라/포인트클라우드 데이터 수신중"];
      }
      return ["BOOTING", "ROS 데이터 수신 대기중"];
    }

    function updateMissionAndChips(data) {
      const ts = data.timestamps || {};
      const counts = data.counts || {};

      const rgbOk = isFresh(ts.rgb, 3);
      const pcOk = isFresh(ts.pointcloud, 3);
      const mapOk = isFresh(ts.map, 8);
      const baseOk = isFresh(ts.safety, 3) || isFresh(ts.person, 3) || isFresh(ts.cmd_out, 3) || isFresh(ts.odom, 3);
      const vlaOk = isFresh(ts.vla, 5) || isFresh(ts.vla_left, 5) || isFresh(ts.vla_right, 5);

      setChip("chip-rgb", rgbOk ? "ok" : "bad", rgbOk ? shortAge(ts.rgb) : "WAIT");
      setChip("chip-pc", pcOk ? "ok" : "bad", pcOk ? shortAge(ts.pointcloud) : "WAIT");
      setChip("chip-map", mapOk ? "ok" : "warn", mapOk ? shortAge(ts.map) : "WAIT");
      setChip("chip-base", baseOk ? "ok" : "warn", baseOk ? shortAge(Math.max(ts.safety || 0, ts.person || 0, ts.cmd_out || 0, ts.odom || 0)) : "WAIT");
      setChip("chip-vla", vlaOk ? "ok" : "warn", vlaOk ? shortAge(Math.max(ts.vla || 0, ts.vla_left || 0, ts.vla_right || 0)) : "WAIT");

      const [mission, desc] = inferMissionState(data);
      $("mission-state").textContent = mission;
      $("mission-desc").textContent = desc;

      $("rgb-count").textContent = counts.rgb || 0;
      $("map-count").textContent = counts.map || 0;

      const pc = data.point_cloud || {};
      const pcCount = pc.count || 0;
      $("pc-count").textContent = pcCount.toLocaleString();
      $("pc-overlay").textContent = `${pcCount.toLocaleString()} points`;

      const lastTs = Math.max(ts.rgb || 0, ts.pointcloud || 0, ts.map || 0, ts.vlm || 0, ts.safety || 0, ts.vla || 0);
      $("last-update").textContent = shortAge(lastTs);
    }


    const pcViewer = {
      canvas: null,
      gl: null,
      program: null,
      posBuffer: null,
      colorBuffer: null,
      pointCount: 0,
      yaw: 0.0,
      pitch: -0.35,
      distance: 2.4,
      targetX: 0.0,
      targetY: 0.0,
      targetZ: -1.0,
      dragging: false,
      dragButton: 0,
      lastX: 0,
      lastY: 0,
    };

    function mat4Perspective(fovy, aspect, near, far) {
      const f = 1.0 / Math.tan(fovy / 2.0);
      const nf = 1.0 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
      ]);
    }

    function normalize(v) {
      const n = Math.hypot(v[0], v[1], v[2]) || 1.0;
      return [v[0] / n, v[1] / n, v[2] / n];
    }

    function cross(a, b) {
      return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];
    }

    function dot(a, b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    function mat4LookAt(eye, target, up) {
      const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
      const x = normalize(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
      ]);
    }

    function mat4Multiply(a, b) {
      const out = new Float32Array(16);
      for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
          out[col * 4 + row] =
            a[0 * 4 + row] * b[col * 4 + 0] +
            a[1 * 4 + row] * b[col * 4 + 1] +
            a[2 * 4 + row] * b[col * 4 + 2] +
            a[3 * 4 + row] * b[col * 4 + 3];
        }
      }
      return out;
    }

    function makeShader(gl, type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
      }
      return shader;
    }

    function initPointCloudViewer() {
      const canvas = $("pointcloud-canvas");
      if (!canvas) return;
      const gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: false });

      if (!gl) {
        $("pointcloud-placeholder").textContent = "이 브라우저는 WebGL을 지원하지 않습니다";
        return;
      }

      const vs = `
        attribute vec3 a_position;
        attribute vec3 a_color;
        uniform mat4 u_matrix;
        varying vec3 v_color;
        void main() {
          gl_Position = u_matrix * vec4(a_position, 1.0);
          gl_PointSize = 2.0;
          v_color = a_color;
        }
      `;

      const fs = `
        precision mediump float;
        varying vec3 v_color;
        void main() {
          vec2 p = gl_PointCoord - vec2(0.5, 0.5);
          if (dot(p, p) > 0.25) discard;
          gl_FragColor = vec4(v_color, 1.0);
        }
      `;

      const program = gl.createProgram();
      gl.attachShader(program, makeShader(gl, gl.VERTEX_SHADER, vs));
      gl.attachShader(program, makeShader(gl, gl.FRAGMENT_SHADER, fs));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || "program link failed");
      }

      pcViewer.canvas = canvas;
      pcViewer.gl = gl;
      pcViewer.program = program;
      pcViewer.posBuffer = gl.createBuffer();
      pcViewer.colorBuffer = gl.createBuffer();

      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      canvas.addEventListener("mousedown", (e) => {
        pcViewer.dragging = true;
        pcViewer.dragButton = e.button;
        pcViewer.lastX = e.clientX;
        pcViewer.lastY = e.clientY;
        canvas.classList.add("dragging");
      });
      window.addEventListener("mouseup", () => {
        pcViewer.dragging = false;
        canvas.classList.remove("dragging");
      });
      window.addEventListener("mousemove", (e) => {
        if (!pcViewer.dragging) return;
        const dx = e.clientX - pcViewer.lastX;
        const dy = e.clientY - pcViewer.lastY;
        pcViewer.lastX = e.clientX;
        pcViewer.lastY = e.clientY;
        if (pcViewer.dragButton === 2) {
          const panScale = pcViewer.distance * 0.0016;
          pcViewer.targetX -= dx * panScale;
          pcViewer.targetY += dy * panScale;
        } else {
          pcViewer.yaw += dx * 0.006;
          pcViewer.pitch += dy * 0.006;
          pcViewer.pitch = Math.max(-1.45, Math.min(1.45, pcViewer.pitch));
        }
      });
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const scale = Math.exp(e.deltaY * 0.001);
        pcViewer.distance = Math.max(0.25, Math.min(12.0, pcViewer.distance * scale));
      }, { passive: false });

      requestAnimationFrame(renderPointCloud);
    }

    function updatePointCloudViewer(payload) {
      if (!payload || !payload.points || !payload.colors || !pcViewer.gl) return;
      const gl = pcViewer.gl;
      const points = new Float32Array(payload.points);
      const colors = new Float32Array(payload.colors);
      pcViewer.pointCount = Math.floor(points.length / 3);
      gl.bindBuffer(gl.ARRAY_BUFFER, pcViewer.posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, pcViewer.colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
      const ph = $("pointcloud-placeholder");
      if (ph) ph.style.display = pcViewer.pointCount > 0 ? "none" : "block";
    }

    function renderPointCloud() {
      const gl = pcViewer.gl;
      const canvas = pcViewer.canvas;
      if (!gl || !canvas) {
        requestAnimationFrame(renderPointCloud);
        return;
      }
      const w = canvas.clientWidth || 1;
      const h = canvas.clientHeight || 1;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.008, 0.024, 0.070, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);

      if (pcViewer.pointCount > 0) {
        const aspect = canvas.width / Math.max(1, canvas.height);
        const proj = mat4Perspective(Math.PI / 4.0, aspect, 0.03, 30.0);
        const cp = Math.cos(pcViewer.pitch);
        const sp = Math.sin(pcViewer.pitch);
        const sy = Math.sin(pcViewer.yaw);
        const cy = Math.cos(pcViewer.yaw);
        const target = [pcViewer.targetX, pcViewer.targetY, pcViewer.targetZ];
        const eye = [
          target[0] + pcViewer.distance * sy * cp,
          target[1] + pcViewer.distance * sp,
          target[2] + pcViewer.distance * cy * cp,
        ];
        const view = mat4LookAt(eye, target, [0, 1, 0]);
        const matrix = mat4Multiply(proj, view);
        gl.useProgram(pcViewer.program);
        const aPos = gl.getAttribLocation(pcViewer.program, "a_position");
        const aColor = gl.getAttribLocation(pcViewer.program, "a_color");
        const uMatrix = gl.getUniformLocation(pcViewer.program, "u_matrix");
        gl.uniformMatrix4fv(uMatrix, false, matrix);
        gl.bindBuffer(gl.ARRAY_BUFFER, pcViewer.posBuffer);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, pcViewer.colorBuffer);
        gl.enableVertexAttribArray(aColor);
        gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, pcViewer.pointCount);
      }
      requestAnimationFrame(renderPointCloud);
    }

    function updateDashboard(data) {
      setImg("rgb-img", "rgb-placeholder", data.rgb_image);
      updatePointCloudViewer(data.point_cloud);
      setImg("map-img", "map-placeholder", data.map_image);

      $("rgb-ts").textContent = fmtTs(data.timestamps?.rgb, data.counts?.rgb);
      $("depth-ts").textContent = fmtTs(data.timestamps?.pointcloud, data.counts?.pointcloud);
      $("map-ts").textContent = fmtTs(data.timestamps?.map, data.counts?.map);
      $("vlm-ts").textContent = fmtTs(data.timestamps?.vlm || data.timestamps?.stt);
      $("base-ts").textContent = fmtTs(
        data.timestamps?.safety || data.timestamps?.person || data.timestamps?.cmd_out || data.timestamps?.base
      );
      $("arm-ts").textContent = fmtTs(
        data.timestamps?.vla || data.timestamps?.vla_left || data.timestamps?.vla_right || data.timestamps?.arm
      );

      $("stt-text").textContent = safeText(data.stt_text);
      $("robot-speech").textContent = safeText(data.robot_speech || data.vlm_decision?.robot_speech);
      $("vlm-status").textContent = safeText(data.inference_status);

      const d = data.vlm_decision || {};
      const vlmLines = [
        `need_oxygen_mask: ${safeText(d.need_oxygen_mask, "-")}`,
        `selected_task: ${safeText(d.selected_task, "-")}`,
        `adapter_id: ${safeText(d.adapter_id, "-")}`,
        `confidence: ${safeText(d.confidence, "-")}`,
        `mock_action: ${safeText(d.mock_action, "-")}`,
        `latency_sec: ${safeText(d.latency_sec, "-")}`,
        `camera_age_sec: ${safeText(d.camera_age_sec, "-")}`,
        `latest_vad: ${safeText(d.latest_vad, "-")}`,
        `use_vad_gate: ${safeText(d.use_vad_gate, "-")}`,
        `doa_deg: ${safeText(d.doa_deg, "-")}`,
        `doa_age_sec: ${safeText(d.doa_age_sec, "-")}`,
        `live_rgb_topic: ${safeText(d.live_rgb_topic, "-")}`,
        `live_depth_topic: ${safeText(d.live_depth_topic, "-")}`,
        `vlm_rgb_snapshot: ${safeText(d.vlm_input_rgb_topic, "-")}`,
        `vlm_depth_snapshot: ${safeText(d.vlm_input_depth_topic, "-")}`
      ];

      $("vlm-output").textContent = vlmLines.join("\n");
      $("vlm-reason").textContent = safeText(d.reason);

      const cmdLines = [
        "RAW  " + safeText(data.cmd_raw, "-"),
        "OUT  " + safeText(data.cmd_out, "-"),
      ];
      const navLines = [
        "SCAN " + safeText(data.scan_state, "-"),
        "ODOM " + safeText(data.odom_state, "-"),
        "LEGACY " + safeText(data.base_status, "-"),
      ];

      $("person-follow-state").textContent = safeText(data.person_follow_state);
      $("safety-guard-state").textContent = safeText(data.safety_guard_state);
      $("cmd-state").textContent = cmdLines.join("\n");
      $("nav-state").textContent = navLines.join("\n");

      $("vla-status").textContent = safeText(data.vla_status);
      $("vla-left-status").textContent = safeText(data.vla_left_status);
      $("vla-right-status").textContent = safeText(data.vla_right_status);
      $("arm-status").textContent = [
        "TASK_REQUEST " + safeText(data.vla_task_request, "-"),
        "LEGACY " + safeText(data.arm_status, "-"),
      ].join("\n");

      updateMissionAndChips(data);
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        $("ws-dot").classList.add("ok");
        $("ws-state").textContent = "CONNECTED";
        $("chip-ws").classList.remove("bad", "warn");
        $("chip-ws").classList.add("ok");
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
      };

      ws.onclose = () => {
        $("ws-dot").classList.remove("ok");
        $("ws-state").textContent = "DISCONNECTED";
        $("chip-ws").classList.remove("ok", "warn");
        $("chip-ws").classList.add("bad");
        setTimeout(connect, 1000);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    setInterval(() => {
      $("clock").textContent = new Date().toLocaleString();
    }, 500);

    initPointCloudViewer();

    connect();
  </script>
</body>
</html>
"""


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "rgb_image": None,
            "point_cloud": None,
            "map_image": None,
            "stt_text": "",
            "vlm_decision": {},
            "robot_speech": "",
            "inference_status": "대기중",
            "base_status": "데이터 대기중",
            "arm_status": "데이터 대기중",
            "person_follow_state": "데이터 대기중",
            "safety_guard_state": "데이터 대기중",
            "cmd_raw": "데이터 대기중",
            "cmd_out": "데이터 대기중",
            "scan_state": "데이터 대기중",
            "odom_state": "데이터 대기중",
            "vla_status": "데이터 대기중",
            "vla_left_status": "데이터 대기중",
            "vla_right_status": "데이터 대기중",
            "vla_task_request": "데이터 대기중",
            "timestamps": {},
            "counts": {
                "rgb": 0,
                "pointcloud": 0,
                "map": 0,
            },
        }

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            for key, value in kwargs.items():
                self.data[key] = value

    def update_timestamp(self, key: str) -> None:
        with self.lock:
            self.data.setdefault("timestamps", {})[key] = time.time()

    def increment_count(self, key: str) -> None:
        with self.lock:
            self.data.setdefault("counts", {})
            self.data["counts"][key] = int(self.data["counts"].get(key, 0)) + 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data, ensure_ascii=False))


STATE = SharedState()
app = FastAPI()


def make_qos(reliability: str, depth: int) -> QoSProfile:
    if reliability == "best_effort":
        rel = ReliabilityPolicy.BEST_EFFORT
    else:
        rel = ReliabilityPolicy.RELIABLE

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=rel,
    )


def encode_image_to_data_url(
    image_bgr: Optional[np.ndarray],
    ext: str = ".jpg",
    quality: int = 80,
) -> Optional[str]:
    if image_bgr is None:
        return None

    params = []
    if ext.lower() in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    ok, encoded = cv2.imencode(ext, image_bgr, params)
    if not ok:
        return None

    mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_msg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("bgr8", "rgb8"):
            channels = 3
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgb8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            return arr.copy()

        if encoding in ("bgra8", "rgba8"):
            channels = 4
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step // channels
            arr = arr.reshape((height, row_pixels, channels))[:, :width, :]

            if encoding == "rgba8":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

            return arr.copy()

        if encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            arr = arr.reshape((height, row_pixels))[:, :width]
            return cv2.cvtColor(arr.copy(), cv2.COLOR_GRAY2BGR)

        if encoding in ("16uc1", "mono16", "32fc1"):
            return depth_msg_to_colormap(msg)

    except Exception:
        return None

    return None


def depth_msg_to_colormap(msg: Image) -> Optional[np.ndarray]:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if height <= 0 or width <= 0:
        return None

    raw = bytes(msg.data)

    try:
        if encoding in ("16uc1", "mono16"):
            arr = np.frombuffer(raw, dtype=np.uint16)
            row_pixels = step // np.dtype(np.uint16).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)
            valid = depth[depth > 0]

        elif encoding == "32fc1":
            arr = np.frombuffer(raw, dtype=np.float32)
            row_pixels = step // np.dtype(np.float32).itemsize
            depth = arr.reshape((height, row_pixels))[:, :width].astype(np.float32)
            valid = depth[np.isfinite(depth) & (depth > 0)]

        elif encoding in ("mono8", "8uc1"):
            arr = np.frombuffer(raw, dtype=np.uint8)
            row_pixels = step
            gray = arr.reshape((height, row_pixels))[:, :width]
            return cv2.applyColorMap(gray.copy(), cv2.COLORMAP_JET)

        else:
            return image_msg_to_bgr(msg)

        if valid.size == 0:
            gray = np.zeros((height, width), dtype=np.uint8)
        else:
            lo = float(np.percentile(valid, 5))
            hi = float(np.percentile(valid, 95))

            if hi <= lo:
                hi = lo + 1.0

            clipped = np.clip(depth, lo, hi)
            norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
            norm[depth <= 0] = 0
            gray = norm

        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return colored

    except Exception:
        return None


def occupancy_grid_to_png_data_url(msg: OccupancyGrid) -> Optional[str]:
    width = int(msg.info.width)
    height = int(msg.info.height)

    if width <= 0 or height <= 0:
        return None

    try:
        grid = np.array(msg.data, dtype=np.int16).reshape((height, width))

        img = np.zeros((height, width), dtype=np.uint8)
        img[grid < 0] = 127
        img[grid == 0] = 255
        img[(grid > 0) & (grid < 100)] = 180
        img[grid >= 100] = 0

        img = np.flipud(img)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        scale = max(1, min(4, 640 // max(1, width)))
        bgr = cv2.resize(
            bgr,
            (width * scale, height * scale),
            interpolation=cv2.INTER_NEAREST,
        )

        return encode_image_to_data_url(bgr, ext=".png")

    except Exception:
        return None



_POINT_FIELD_DTYPES = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


def pointcloud2_to_payload(
    msg: PointCloud2,
    max_points: int = 30000,
    max_range_m: float = 5.0,
) -> Optional[Dict[str, Any]]:
    if msg.width <= 0 or msg.height <= 0 or msg.point_step <= 0:
        return None

    field_map = {field.name: field for field in msg.fields}
    if not {"x", "y", "z"}.issubset(field_map.keys()):
        return None

    names = []
    formats = []
    offsets = []
    endian = ">" if msg.is_bigendian else "<"

    for field in msg.fields:
        base = _POINT_FIELD_DTYPES.get(int(field.datatype))
        if base is None:
            continue
        base = base.newbyteorder(endian)
        count = max(1, int(field.count))
        names.append(field.name)
        formats.append(base if count == 1 else (base, count))
        offsets.append(int(field.offset))

    try:
        dtype = np.dtype({
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": int(msg.point_step),
        })
        total_points = int(msg.width) * int(msg.height)
        arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=total_points)
        x = np.asarray(arr["x"], dtype=np.float32).reshape(-1)
        y = np.asarray(arr["y"], dtype=np.float32).reshape(-1)
        z = np.asarray(arr["z"], dtype=np.float32).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid &= z > 0.02
        if max_range_m > 0:
            valid &= np.sqrt(x * x + y * y + z * z) <= float(max_range_m)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return {"points": [], "colors": [], "count": 0, "frame_id": msg.header.frame_id, "source_points": total_points}
        if indices.size > max_points:
            step = int(math.ceil(indices.size / float(max_points)))
            indices = indices[::step][:max_points]
        pts = np.column_stack((x[indices], -y[indices], -z[indices])).astype(np.float32)
        colors = None
        if "rgb" in arr.dtype.names:
            rgb_raw = np.asarray(arr["rgb"][indices]).reshape(-1)
            if rgb_raw.dtype == np.float32:
                rgb_uint = rgb_raw.copy().view(np.uint32)
            else:
                rgb_uint = rgb_raw.astype(np.uint32, copy=False)
            colors = np.column_stack((((rgb_uint >> 16) & 255), ((rgb_uint >> 8) & 255), (rgb_uint & 255))).astype(np.float32) / 255.0
        elif "rgba" in arr.dtype.names:
            rgba_raw = np.asarray(arr["rgba"][indices]).reshape(-1)
            if rgba_raw.dtype == np.float32:
                rgba_uint = rgba_raw.copy().view(np.uint32)
            else:
                rgba_uint = rgba_raw.astype(np.uint32, copy=False)
            colors = np.column_stack((((rgba_uint >> 16) & 255), ((rgba_uint >> 8) & 255), (rgba_uint & 255))).astype(np.float32) / 255.0
        elif {"r", "g", "b"}.issubset(arr.dtype.names):
            colors = np.column_stack((np.asarray(arr["r"][indices]).reshape(-1), np.asarray(arr["g"][indices]).reshape(-1), np.asarray(arr["b"][indices]).reshape(-1))).astype(np.float32) / 255.0
        if colors is None:
            depth = np.clip(z[indices], 0.0, max(0.1, float(max_range_m)))
            norm = depth / max(0.1, float(max_range_m))
            colors = np.column_stack((0.25 + 0.75 * (1.0 - norm), 0.55 + 0.35 * norm, 1.0 * norm)).astype(np.float32)
        return {"points": np.round(pts, 4).reshape(-1).tolist(), "colors": np.round(colors, 4).reshape(-1).tolist(), "count": int(pts.shape[0]), "source_points": total_points, "frame_id": msg.header.frame_id}
    except Exception:
        return None


def parse_json_string_or_raw(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": text}


def pretty_json_string_or_raw(text: str) -> str:
    parsed = parse_json_string_or_raw(text)
    if not parsed:
        return "대기중"
    if set(parsed.keys()) == {"raw"}:
        return str(parsed["raw"])
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def twist_to_text(msg: Twist) -> str:
    return (
        f"linear.x={msg.linear.x:+.3f}, linear.y={msg.linear.y:+.3f}, linear.z={msg.linear.z:+.3f}\n"
        f"angular.x={msg.angular.x:+.3f}, angular.y={msg.angular.y:+.3f}, angular.z={msg.angular.z:+.3f}"
    )


def odom_to_text(msg: Odometry) -> str:
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )
    yaw_deg = math.degrees(yaw)
    v = msg.twist.twist.linear
    w = msg.twist.twist.angular
    return (
        f"pos x={p.x:+.3f}, y={p.y:+.3f}, yaw={yaw_deg:+.1f}deg\n"
        f"vel x={v.x:+.3f}, y={v.y:+.3f}, wz={w.z:+.3f}"
    )


def scan_to_text(msg: LaserScan) -> str:
    values = np.asarray(msg.ranges, dtype=np.float32)
    finite = values[np.isfinite(values)]
    valid = finite[(finite > max(0.0, float(msg.range_min))) & (finite < float(msg.range_max))]
    if valid.size == 0:
        return "valid_range=0"
    return (
        f"min={float(np.min(valid)):.3f}m, mean={float(np.mean(valid)):.3f}m, "
        f"valid={int(valid.size)}/{len(msg.ranges)}"
    )


class ZeriDashboardNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("zeri_dashboard_server_node")

        self.args = args

        image_qos = make_qos(args.image_qos, args.image_qos_depth)
        text_qos = make_qos("reliable", 10)
        map_qos = make_qos(args.map_qos, 5)

        self.create_subscription(Image, args.rgb_topic, self.rgb_callback, image_qos)
        self.create_subscription(PointCloud2, args.pointcloud_topic, self.pointcloud_callback, image_qos)
        self.create_subscription(OccupancyGrid, args.map_topic, self.map_callback, map_qos)

        self.create_subscription(String, args.stt_topic, self.stt_callback, text_qos)
        self.create_subscription(String, args.vlm_decision_topic, self.vlm_decision_callback, text_qos)
        self.create_subscription(String, args.robot_speech_topic, self.robot_speech_callback, text_qos)
        self.create_subscription(String, args.inference_status_topic, self.inference_status_callback, text_qos)

        self.create_subscription(String, args.base_status_topic, self.base_status_callback, text_qos)
        self.create_subscription(String, args.arm_status_topic, self.arm_status_callback, text_qos)

        self.create_subscription(String, args.person_state_topic, self.person_state_callback, text_qos)
        self.create_subscription(String, args.safety_state_topic, self.safety_state_callback, text_qos)
        self.create_subscription(Twist, args.cmd_raw_topic, self.cmd_raw_callback, text_qos)
        self.create_subscription(Twist, args.cmd_out_topic, self.cmd_out_callback, text_qos)
        self.create_subscription(LaserScan, args.scan_topic, self.scan_callback, image_qos)
        self.create_subscription(Odometry, args.odom_topic, self.odom_callback, text_qos)

        self.create_subscription(String, args.vla_status_topic, self.vla_status_callback, text_qos)
        self.create_subscription(String, args.vla_left_status_topic, self.vla_left_status_callback, text_qos)
        self.create_subscription(String, args.vla_right_status_topic, self.vla_right_status_callback, text_qos)
        self.create_subscription(String, args.vla_task_request_topic, self.vla_task_request_callback, text_qos)

        self.get_logger().info("Ze-Ri Dashboard subscriptions:")
        self.get_logger().info(f"  RGB:              {args.rgb_topic}")
        self.get_logger().info(f"  3D PointCloud:    {args.pointcloud_topic}")
        self.get_logger().info(f"  Map:              {args.map_topic}")
        self.get_logger().info(f"  STT:              {args.stt_topic}")
        self.get_logger().info(f"  VLM decision:     {args.vlm_decision_topic}")
        self.get_logger().info(f"  Robot speech:     {args.robot_speech_topic}")
        self.get_logger().info(f"  Inference status: {args.inference_status_topic}")
        self.get_logger().info(f"  Base status:      {args.base_status_topic}")
        self.get_logger().info(f"  Arm status:       {args.arm_status_topic}")
        self.get_logger().info(f"  Person state:     {args.person_state_topic}")
        self.get_logger().info(f"  Safety state:     {args.safety_state_topic}")
        self.get_logger().info(f"  cmd raw/out:      {args.cmd_raw_topic} -> {args.cmd_out_topic}")
        self.get_logger().info(f"  Scan/Odom:        {args.scan_topic} / {args.odom_topic}")
        self.get_logger().info(f"  VLA status:       {args.vla_status_topic}")
        self.get_logger().info(f"  VLA left/right:   {args.vla_left_status_topic} / {args.vla_right_status_topic}")
        self.get_logger().info(f"  VLA request:      {args.vla_task_request_topic}")
        self.get_logger().info(f"  Image QoS:        {args.image_qos}")

        self.mock_start_time = time.time()

        if args.mock_status:
            self.mock_status_timer = self.create_timer(0.5, self.mock_status_callback)
            self.get_logger().info("  Mock status:      enabled")
        else:
            self.get_logger().info("  Mock status:      disabled")

    def rgb_callback(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        data_url = encode_image_to_data_url(
            bgr,
            ext=".jpg",
            quality=self.args.jpeg_quality,
        )

        if data_url:
            STATE.update(rgb_image=data_url)
            STATE.update_timestamp("rgb")
            STATE.increment_count("rgb")

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        payload = pointcloud2_to_payload(
            msg,
            max_points=int(self.args.pointcloud_max_points),
            max_range_m=float(self.args.pointcloud_max_range_m),
        )

        if payload is not None:
            STATE.update(point_cloud=payload)
            STATE.update_timestamp("pointcloud")
            STATE.increment_count("pointcloud")

    def map_callback(self, msg: OccupancyGrid) -> None:
        data_url = occupancy_grid_to_png_data_url(msg)

        if data_url:
            STATE.update(map_image=data_url)
            STATE.update_timestamp("map")
            STATE.increment_count("map")

    def stt_callback(self, msg: String) -> None:
        STATE.update(stt_text=msg.data)
        STATE.update_timestamp("stt")

    def vlm_decision_callback(self, msg: String) -> None:
        parsed = parse_json_string_or_raw(msg.data)

        if "stt_text" in parsed:
            STATE.update(stt_text=str(parsed.get("stt_text", "")))

        if "robot_speech" in parsed:
            STATE.update(robot_speech=str(parsed.get("robot_speech", "")))

        STATE.update(vlm_decision=parsed)
        STATE.update_timestamp("vlm")

    def robot_speech_callback(self, msg: String) -> None:
        STATE.update(robot_speech=msg.data)
        STATE.update_timestamp("vlm")

    def inference_status_callback(self, msg: String) -> None:
        STATE.update(inference_status=msg.data)
        STATE.update_timestamp("vlm")

    def base_status_callback(self, msg: String) -> None:
        STATE.update(base_status=msg.data)
        STATE.update_timestamp("base")

    def arm_status_callback(self, msg: String) -> None:
        STATE.update(arm_status=msg.data)
        STATE.update_timestamp("arm")

    def person_state_callback(self, msg: String) -> None:
        STATE.update(person_follow_state=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("person")

    def safety_state_callback(self, msg: String) -> None:
        STATE.update(safety_guard_state=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("safety")

    def cmd_raw_callback(self, msg: Twist) -> None:
        STATE.update(cmd_raw=twist_to_text(msg))
        STATE.update_timestamp("cmd_raw")

    def cmd_out_callback(self, msg: Twist) -> None:
        STATE.update(cmd_out=twist_to_text(msg))
        STATE.update_timestamp("cmd_out")

    def scan_callback(self, msg: LaserScan) -> None:
        STATE.update(scan_state=scan_to_text(msg))
        STATE.update_timestamp("scan")

    def odom_callback(self, msg: Odometry) -> None:
        STATE.update(odom_state=odom_to_text(msg))
        STATE.update_timestamp("odom")

    def vla_status_callback(self, msg: String) -> None:
        STATE.update(vla_status=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("vla")

    def vla_left_status_callback(self, msg: String) -> None:
        STATE.update(vla_left_status=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("vla_left")

    def vla_right_status_callback(self, msg: String) -> None:
        STATE.update(vla_right_status=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("vla_right")

    def vla_task_request_callback(self, msg: String) -> None:
        STATE.update(vla_task_request=pretty_json_string_or_raw(msg.data))
        STATE.update_timestamp("vla")

    def mock_status_callback(self) -> None:
        t = time.time() - self.mock_start_time

        base_modes = [
            "IDLE",
            "VOICE_TRACKING",
            "APPROACHING_TARGET",
            "OBSTACLE_GUARD_ACTIVE",
        ]
        base_mode = base_modes[int(t // 6.0) % len(base_modes)]

        vx = 0.10 + 0.04 * math.sin(t * 0.7)
        vy = 0.02 * math.sin(t * 0.5)
        wz = 0.18 * math.sin(t * 0.4)

        left_rpm = 22.0 + 5.0 * math.sin(t * 0.9)
        rear_rpm = 18.0 + 4.0 * math.sin(t * 0.8)
        right_rpm = 22.0 + 5.0 * math.sin(t * 1.0 + 0.5)

        base_battery = max(0.0, 96.0 - 0.015 * t)

        base_status = (
            f"MODE: {base_mode}\n"
            f"cmd_vel: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}\n"
            f"rpm: L={left_rpm:.1f}, Rr={rear_rpm:.1f}, R={right_rpm:.1f}\n"
            f"battery={base_battery:.1f}% | obstacle_guard=ON | source=MOCK"
        )

        arm_states = [
            "HOME",
            "READY",
            "MASK_PICK_PREP",
            "HOLDING_OXYGEN_MASK",
        ]
        arm_state = arm_states[int(t // 7.0) % len(arm_states)]

        shoulder_pan = 5.0 * math.sin(t * 0.4)
        shoulder_lift = -35.0 + 4.0 * math.sin(t * 0.5)
        elbow_flex = 42.0 + 6.0 * math.sin(t * 0.6)
        wrist_flex = 80.0 + 5.0 * math.sin(t * 0.7)
        wrist_roll = 120.0 + 8.0 * math.sin(t * 0.8)
        gripper = 35.0 + 15.0 * math.sin(t * 0.9)

        arm_status = (
            f"STATE: {arm_state}\n"
            f"policy: idle_lora | SO-101: MOCK\n"
            f"joints_deg: pan={shoulder_pan:.1f}, lift={shoulder_lift:.1f}, "
            f"elbow={elbow_flex:.1f}\n"
            f"wrist_f={wrist_flex:.1f}, wrist_r={wrist_roll:.1f}, "
            f"gripper={gripper:.1f}%\n"
            f"source=MOCK"
        )

        STATE.update(base_status=base_status, arm_status=arm_status)
        STATE.update_timestamp("base")
        STATE.update_timestamp("arm")


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        INDEX_HTML,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            await websocket.send_text(json.dumps(STATE.snapshot(), ensure_ascii=False))
            await asyncio.sleep(0.25)

    except WebSocketDisconnect:
        return

    except AssertionError:
        # Browser tab closed/reloaded while a large PointCloud JSON frame was being sent.
        return

    except (RuntimeError, ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ze-Ri ROS2 Web Dashboard")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=75)

    parser.add_argument("--image-qos", choices=["reliable", "best_effort"], default="best_effort")
    parser.add_argument("--map-qos", choices=["reliable", "best_effort"], default="reliable")
    parser.add_argument("--image-qos-depth", type=int, default=10)

    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--pointcloud-topic", default="/camera/camera/depth/color/points")
    parser.add_argument("--pointcloud-max-points", type=int, default=30000)
    parser.add_argument("--pointcloud-max-range-m", type=float, default=5.0)
    parser.add_argument("--map-topic", default="/map")

    parser.add_argument("--stt-topic", default="/stt/text")
    parser.add_argument("--vlm-decision-topic", default="/zeri/vlm/decision")
    parser.add_argument("--robot-speech-topic", default="/zeri/vlm/robot_speech")
    parser.add_argument("--inference-status-topic", default="/zeri/vlm/inference_status")

    parser.add_argument("--base-status-topic", default="/zeri/mobile_base/status")
    parser.add_argument("--arm-status-topic", default="/zeri/arm/status")

    parser.add_argument("--person-state-topic", default="/zeri/person_follow/state")
    parser.add_argument("--safety-state-topic", default="/zeri/safety_guard/state")
    parser.add_argument("--cmd-raw-topic", default="/cmd_vel_raw")
    parser.add_argument("--cmd-out-topic", default="/cmd_vel")
    parser.add_argument("--scan-topic", default="/scan_front")
    parser.add_argument("--odom-topic", default="/odom")

    parser.add_argument("--vla-status-topic", default="/zeri/vla/status")
    parser.add_argument("--vla-left-status-topic", default="/zeri/vla/left/status")
    parser.add_argument("--vla-right-status-topic", default="/zeri/vla/right/status")
    parser.add_argument("--vla-task-request-topic", default="/zeri/vla/task_request")

    parser.add_argument(
        "--mock-status",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show temporary mock mobile-base and arm status on dashboard.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    if not hasattr(args, "pointcloud_topic"):
        args.pointcloud_topic = "/camera/camera/depth/color/points"
    if not hasattr(args, "pointcloud_max_points"):
        args.pointcloud_max_points = 30000
    if not hasattr(args, "pointcloud_max_range_m"):
        args.pointcloud_max_range_m = 5.0

    node = ZeriDashboardNode(args)

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    spin_thread.start()

    print("")
    print("====================================================")
    print(" Ze-Ri ROS2 Dashboard")
    print(f" URL: http://{args.host}:{args.port}")
    print(" Remote:")
    print(f" http://<ROBOT_IP>:{args.port}")
    print("====================================================")
    print("")

    try:
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.run()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
