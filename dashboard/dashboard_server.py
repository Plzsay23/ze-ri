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
      --bg: #f3f6fb;
      --card: #0f63d8;
      --card2: #0b56c4;
      --border: #062b63;
      --text: #ffffff;
      --muted: #dbeafe;
      --dark-panel: #071f49;
      --dark: #0f172a;
    }

    * {
      box-sizing: border-box;
    }

    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
    }

    body {
      background: var(--bg);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--dark);
      overflow: hidden;
    }

    header {
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 10px;
      background: #0f172a;
      color: white;
      border-bottom: 3px solid #2563eb;
    }

    header .title {
      font-size: 16px;
      font-weight: 900;
      letter-spacing: 0.2px;
    }

    header .status {
      font-size: 12px;
      font-weight: 700;
      color: #bfdbfe;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #ef4444;
      display: inline-block;
    }

    .dot.ok {
      background: #22c55e;
    }

    .grid {
      width: 100vw;
      height: calc(100vh - 36px);
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      grid-template-rows: minmax(0, 45fr) minmax(0, 27.5fr) minmax(0, 27.5fr);
      grid-template-areas:
        "rgb  rgb  rgb  rgb  rgb  rgb  depth depth depth depth depth depth"
        "text text text map  map  map  map   map   base  base  base  base"
        "text text text map  map  map  map   map   arm   arm   arm   arm";
      gap: 10px;
      padding: 10px;
      margin: 0;
    }

    .card {
      background: linear-gradient(135deg, var(--card), var(--card2));
      border: 3px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 4px 10px rgba(15, 23, 42, 0.24);
      color: var(--text);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
    }

    .rgb-card { grid-area: rgb; }
    .depth-card { grid-area: depth; }
    .text-card { grid-area: text; }
    .map-card { grid-area: map; }
    .base-card { grid-area: base; }
    .arm-card { grid-area: arm; }

    .card-title {
      height: 32px;
      flex: 0 0 32px;
      padding: 6px 10px;
      font-size: 15px;
      font-weight: 900;
      background: rgba(0, 0, 0, 0.17);
      border-bottom: 1px solid rgba(255, 255, 255, 0.18);
      display: flex;
      align-items: center;
      justify-content: space-between;
      min-width: 0;
    }

    .card-title small {
      font-size: 10px;
      color: var(--muted);
      font-weight: 800;
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
      background: var(--dark-panel);
      overflow: hidden;
    }

    .media-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: auto;
      background: var(--dark-panel);
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
      background: radial-gradient(circle at center, #0f172a 0%, #020617 68%);
      cursor: grab;
    }

    .pointcloud-wrap canvas.dragging {
      cursor: grabbing;
    }

    .viewer-hint {
      position: absolute;
      left: 10px;
      bottom: 8px;
      padding: 5px 7px;
      border-radius: 6px;
      background: rgba(2, 6, 23, 0.72);
      color: #bfdbfe;
      font-size: 10px;
      font-weight: 800;
      pointer-events: none;
      white-space: nowrap;
    }

    .placeholder {
      color: #bfdbfe;
      font-size: 20px;
      font-weight: 900;
      text-align: center;
      padding: 14px;
    }

    .text-panel {
      flex: 1 1 auto;
      min-height: 0;
      padding: 8px;
      overflow: auto;
      background: rgba(5, 20, 48, 0.20);
    }

    .metric {
      margin-bottom: 7px;
      padding: 7px 9px;
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 8px;
    }

    .metric-label {
      color: #dbeafe;
      font-size: 11px;
      font-weight: 900;
      margin-bottom: 4px;
    }

    .metric-value {
      color: white;
      font-size: 15px;
      font-weight: 800;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .metric-value.small {
      font-size: 11px;
      line-height: 1.30;
      font-weight: 700;
    }

    .status-big {
      flex: 1 1 auto;
      min-height: 0;
      padding: 8px;
      display: block;
      background: rgba(5, 20, 48, 0.15);
      overflow: auto;
    }

    .status-line {
      padding: 8px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.18);
      overflow: auto;
      margin-bottom: 7px;
    }

    .status-line:last-child {
      margin-bottom: 0;
    }

    .status-line.compact-line {
      min-height: 0;
    }

    .status-line .label {
      font-size: 11px;
      font-weight: 900;
      color: #dbeafe;
      margin-bottom: 5px;
    }

    .status-line .value {
      font-size: 12px;
      line-height: 1.23;
      font-weight: 800;
      color: white;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }

    @media (max-width: 1200px) {
      body {
        overflow-y: auto;
      }

      .grid {
        height: auto;
        min-height: calc(100vh - 36px);
        grid-template-columns: 1fr;
        grid-template-rows:
          320px
          320px
          360px
          360px
          260px
          260px;
        grid-template-areas:
          "rgb"
          "depth"
          "text"
          "map"
          "base"
          "arm";
      }
    }
  </style>
</head>

<body>
  <header>
    <div class="title">Ze-Ri ROS2 Autonomy / VLA Dashboard</div>
    <div class="status">
      <span><span id="ws-dot" class="dot"></span> <span id="ws-state">DISCONNECTED</span></span>
      <span>|</span>
      <span id="clock">-</span>
    </div>
  </header>

  <main class="grid">
    <section class="card rgb-card">
      <div class="card-title">
        <span>RGB 채널</span>
        <small id="rgb-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="rgb-img" alt="RGB Stream" style="display:none" />
        <div id="rgb-placeholder" class="placeholder">RGB 데이터 대기중</div>
      </div>
    </section>

    <section class="card depth-card">
      <div class="card-title">
        <span>3D Map / PointCloud</span>
        <small id="depth-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <div class="pointcloud-wrap">
          <canvas id="pointcloud-canvas"></canvas>
          <div id="pointcloud-placeholder" class="placeholder">PointCloud 데이터 대기중</div>
          <div class="viewer-hint">좌클릭 회전 · 우클릭 이동 · 휠 줌</div>
        </div>
      </div>
    </section>

    <section class="card text-card">
      <div class="card-title">
        <span>STT / VLM 출력</span>
        <small id="vlm-ts">대기중</small>
      </div>
      <div class="text-panel">
        <div class="metric">
          <div class="metric-label">STT 입력값</div>
          <div id="stt-text" class="metric-value">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">TTS 출력</div>
          <div id="robot-speech" class="metric-value">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 상태</div>
          <div id="vlm-status" class="metric-value small">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 출력</div>
          <div id="vlm-output" class="metric-value small">대기중</div>
        </div>

        <div class="metric">
          <div class="metric-label">VLM 출력 근거</div>
          <div id="vlm-reason" class="metric-value small">대기중</div>
        </div>
      </div>
    </section>

    <section class="card map-card">
      <div class="card-title">
        <span>2D Map</span>
        <small id="map-ts">대기중</small>
      </div>
      <div class="media-wrap">
        <img id="map-img" alt="2D Map" style="display:none" />
        <div id="map-placeholder" class="placeholder">Map 데이터 대기중</div>
      </div>
    </section>

    <section class="card base-card">
      <div class="card-title">
        <span>자율주행 / 베이스 상태</span>
        <small id="base-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">Person Follow</div>
          <div id="person-follow-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
          <div class="label">LiDAR + Depth Safety Guard</div>
          <div id="safety-guard-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
          <div class="label">cmd_vel_raw → cmd_vel</div>
          <div id="cmd-state" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
          <div class="label">Scan / Odom</div>
          <div id="nav-state" class="value">데이터 대기중</div>
        </div>
      </div>
    </section>

    <section class="card arm-card">
      <div class="card-title">
        <span>VLA / 로봇팔 상태</span>
        <small id="arm-ts">대기중</small>
      </div>
      <div class="status-big">
        <div class="status-line">
          <div class="label">VLA Router</div>
          <div id="vla-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
          <div class="label">Left Arm</div>
          <div id="vla-left-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
          <div class="label">Right Arm</div>
          <div id="vla-right-status" class="value">데이터 대기중</div>
        </div>
        <div class="status-line compact-line">
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
      const gl = canvas.getContext("webgl", {
        antialias: true,
        preserveDrawingBuffer: false,
      });

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
          gl_PointSize = 2.2;
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
      if (pcViewer.pointCount > 0) {
        ph.style.display = "none";
      } else {
        ph.style.display = "block";
      }
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
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        $("ws-dot").classList.add("ok");
        $("ws-state").textContent = "CONNECTED";
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
      };

      ws.onclose = () => {
        $("ws-dot").classList.remove("ok");
        $("ws-state").textContent = "DISCONNECTED";
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
    max_points: int = 6000,
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
            return {
                "points": [],
                "colors": [],
                "count": 0,
                "frame_id": msg.header.frame_id,
                "source_points": total_points,
            }

        if indices.size > max_points:
            step = int(math.ceil(indices.size / float(max_points)))
            indices = indices[::step][:max_points]

        # RealSense optical frame: x=right, y=down, z=forward.
        # Viewer frame: x=right, y=up, z=forward-into-screen.
        pts = np.column_stack((x[indices], -y[indices], -z[indices])).astype(np.float32)

        colors = None

        if "rgb" in arr.dtype.names:
            rgb_raw = np.asarray(arr["rgb"][indices]).reshape(-1)
            if rgb_raw.dtype == np.float32:
                rgb_uint = rgb_raw.copy().view(np.uint32)
            else:
                rgb_uint = rgb_raw.astype(np.uint32, copy=False)

            colors = np.column_stack((
                ((rgb_uint >> 16) & 255),
                ((rgb_uint >> 8) & 255),
                (rgb_uint & 255),
            )).astype(np.float32) / 255.0

        elif "rgba" in arr.dtype.names:
            rgba_raw = np.asarray(arr["rgba"][indices]).reshape(-1)
            if rgba_raw.dtype == np.float32:
                rgba_uint = rgba_raw.copy().view(np.uint32)
            else:
                rgba_uint = rgba_raw.astype(np.uint32, copy=False)

            colors = np.column_stack((
                ((rgba_uint >> 16) & 255),
                ((rgba_uint >> 8) & 255),
                (rgba_uint & 255),
            )).astype(np.float32) / 255.0

        elif {"r", "g", "b"}.issubset(arr.dtype.names):
            colors = np.column_stack((
                np.asarray(arr["r"][indices]).reshape(-1),
                np.asarray(arr["g"][indices]).reshape(-1),
                np.asarray(arr["b"][indices]).reshape(-1),
            )).astype(np.float32) / 255.0

        if colors is None:
            depth = np.clip(z[indices], 0.0, max(0.1, float(max_range_m)))
            norm = depth / max(0.1, float(max_range_m))
            colors = np.column_stack((
                0.25 + 0.75 * (1.0 - norm),
                0.55 + 0.35 * norm,
                1.0 * norm,
            )).astype(np.float32)

        return {
            "points": np.round(pts, 4).reshape(-1).tolist(),
            "colors": np.round(colors, 4).reshape(-1).tolist(),
            "count": int(pts.shape[0]),
            "source_points": total_points,
            "frame_id": msg.header.frame_id,
        }

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
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
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
    parser.add_argument("--pointcloud-max-points", type=int, default=6000)
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
        args.pointcloud_max_points = 6000
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
