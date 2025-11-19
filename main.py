import sys
import threading
import queue
import time
import inspect
import math
import ast
import pygame
from pygame import Rect
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

###############################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>  PASTE YOUR TEST CODE HERE  <<<<<<<<<<<<<<<<<<<<<<<<
###############################################################################
# - 여기에 테스트 코드를 그대로 붙여 넣으세요.
# - 선택사항: run_target()/main()/run 중 하나를 정의하면 그 함수를 호출합니다.
# - 아무 엔트리 함수가 없으면, 모듈 최상위 코드만 실행/추적합니다.
TEST_CODE = r"""
# 합병 정렬
def f(low, high):
    if low >= high: # 리스트 인덱스 개수가 0개거나 음수 이면
        return
    
    mid = (low + high)//2

    f(low, mid)
    f(mid+1, high)
    
    i = low
    j = mid + 1

    for k in range(low, high+1):
        if j > high: # 두 블록의 최댓 값 비교
            B[k] = A[i]
            i = i + 1
        elif i>mid:
            B[k]=A[j]
            j += 1
        elif A[i]<=A[j]:
            B[k]=A[i]
            i+=1
        else:
            B[k]=A[j]
            j+=1
    
    for k in range(low, high+1):
        A[k]=B[k]
    return


#A = list(map(int, input().split()))
A = [8,4,1,5,7,3,2,6]
B = [0]*len(A)

f(0, len(A)-1) # 인덱스 번호(0 ~ len(A)-1)

print(A)
"""
# 소스의 "가짜 파일명"(UI/트레이서 식별용). 바꿔도 되지만 확장자는 .py 권장.
TEST_FILENAME = "backtrack.py"

###############################################################################
# Utility
###############################################################################

def safe_repr(v: Any, maxlen: int = 60) -> str:
    try:
        s = repr(v)
    except Exception:
        s = f"<{type(v).__name__}>"
    if len(s) > maxlen:
        s = s[:maxlen-3] + "..."
    return s

def get_stack_depth_in_file(frame) -> int:
    depth = 0
    f = frame
    target = frame.f_code.co_filename
    while f is not None and f.f_code.co_filename == target:
        depth += 1
        f = f.f_back
    return depth

def wrap_text_to_lines(font: pygame.font.Font, text: str, max_w: int) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        test = w if not cur else cur + " " + w
        wpx, _ = font.size(test)
        if wpx <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

###############################################################################
# Trace collector
###############################################################################

class TraceCollector:
    def __init__(self, event_queue: "queue.Queue[dict]", file_filter: str):
        self.q = event_queue
        self.file_filter = file_filter
        self.prev_locals_by_frame: Dict[Any, Dict[str, Any]] = {}
        self.parent_map: Dict[Any, Optional[str]] = {}

    def _locals_diff(self, frame) -> List[Tuple[str, Any, Any]]:
        prev = self.prev_locals_by_frame.get(frame, {})
        now = dict(frame.f_locals)
        changes: List[Tuple[str, Any, Any]] = []
        for k, v in now.items():
            if k.startswith("__"):
                continue
            if k not in prev or prev[k] != v:
                changes.append((k, prev.get(k, None), v))
        self.prev_locals_by_frame[frame] = now
        return changes

    def _recursion_index(self, frame) -> int:
        name = frame.f_code.co_name
        f = frame; i = 1
        while (
            f.f_back
            and f.f_back.f_code.co_filename == self.file_filter
            and f.f_back.f_code.co_name == name
        ):
            i += 1; f = f.f_back
        return i

    def _disp_name(self, frame) -> str:
        base = frame.f_code.co_name
        idx = self._recursion_index(frame)
        return f"{base}{idx}"

    def _disp_caller(self, frame) -> Optional[str]:
        if frame.f_back and frame.f_back.f_code.co_filename == self.file_filter:
            return self._disp_name(frame.f_back)
        return None

    def tracer(self, frame, event, arg):
        code = frame.f_code
        filename = code.co_filename
        if filename != self.file_filter:
            return self.tracer

        func_disp = self._disp_name(frame)
        caller_disp = self._disp_caller(frame)
        lineno = frame.f_lineno

        if event == "call":
            try:
                args_info = inspect.getargvalues(frame)
                kwargs = {name: frame.f_locals.get(name) for name in args_info.args}
                if args_info.varargs:
                    kwargs[args_info.varargs] = frame.f_locals.get(args_info.varargs)
                if args_info.keywords:
                    kw = frame.f_locals.get(args_info.keywords, {})
                    if isinstance(kw, dict):
                        kwargs.update(kw)
            except Exception:
                kwargs = {}

            self.parent_map[frame] = caller_disp

            self.q.put({
                "type": "call", "func": func_disp, "caller": caller_disp,
                "lineno": lineno, "args": {k: safe_repr(v) for k, v in kwargs.items()},
                "ts": time.time()
            })

        elif event == "return":
            self.q.put({
                "type": "return", "func": func_disp, "caller": self.parent_map.get(frame),
                "value": arg, "ts": time.time()
            })

        elif event == "line":
            for k, old_v, new_v in self._locals_diff(frame):
                self.q.put({
                    "type": "assign", "func": func_disp, "var": k,
                    "old": safe_repr(old_v), "new": safe_repr(new_v),
                    "value": safe_repr(new_v),
                    "lineno": lineno, "ts": time.time()
                })
            self.q.put({"type": "line", "func": func_disp, "lineno": lineno, "ts": time.time()})
        return self.tracer

###############################################################################
# Code map (from string) for the Code Panel
###############################################################################

class CodeMap:
    def __init__(self, *, source: str, pseudo_name: str = "test_code.py"):
        self.file_path = pseudo_name
        self.source = source
        self.lines = self.source.splitlines()
        self.blocks: List[Tuple[int, int, str]] = []
        self._build_blocks()

    def _build_blocks(self):
        try:
            tree = ast.parse(self.source, filename=self.file_path)
        except Exception:
            return
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.If): kind = "if"
            elif isinstance(node, ast.For): kind = "for"
            elif isinstance(node, ast.While): kind = "while"
            if kind and hasattr(node, "lineno"):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if isinstance(start, int) and isinstance(end, int) and end >= start:
                    self.blocks.append((start, end, kind))

###############################################################################
# Pygame UI setup
###############################################################################

WIDTH, HEIGHT = 1400, 840
FPS = 144

MARGIN = 12
LINE_H = 22
CODE_TOP_PADDING = 36

STEP_PAUSE = 1.0
POPUP_TTL = 2.6

TOP_FACTORY_RATIO = 0.55
BOTTOM_CODE_RATIO  = 0.62
RIGHT_SIDEBAR_MIN_W = 300

pygame.init()
FONT = pygame.font.SysFont("consolas", 18)
BIG  = pygame.font.SysFont("consolas", 24, bold=True)
SMALL= pygame.font.SysFont("consolas", 14)

def draw_arrow(surface, start, end, color=(200, 220, 255), width=3, head_len=12):
    x1, y1 = start; x2, y2 = end
    pygame.draw.line(surface, color, start, end, width)
    angle = math.atan2(y2 - y1, x2 - x1)
    hx = x2 - head_len * math.cos(angle - math.pi/6)
    hy = y2 - head_len * math.sin(angle - math.pi/6)
    hx2 = x2 - head_len * math.cos(angle + math.pi/6)
    hy2 = y2 - head_len * math.sin(angle + math.pi/6)
    pygame.draw.polygon(surface, color, [(x2, y2), (hx, hy), (hx2, hy2)])

class Popup:
    def __init__(self, text: str, x: int, y: int, ttl: float = POPUP_TTL):
        self.text = text; self.x = x; self.y = y
        self.ttl = ttl; self.total = ttl
    def update(self, dt: float): self.ttl -= dt
    def draw(self, surface, dy: int = 0):
        alpha = max(0, min(255, int(255 * (self.ttl / self.total))))
        t = FONT.render(self.text, True, (255, 255, 255))
        s = pygame.Surface((t.get_width() + 12, t.get_height() + 8), pygame.SRCALPHA)
        s.fill((20, 24, 32, alpha)); s.blit(t, (6, 4))
        surface.blit(s, (self.x, self.y + dy - (self.total - self.ttl) * 18))

###############################################################################
# Dynamic flow token
###############################################################################

class DynamicFlowToken:
    def __init__(self, text: str, start_fn, end_fn, speed: float = 260.0):
        self.text = text
        self.start_fn = start_fn
        self.end_fn   = end_fn
        self.speed = speed
        self.t = 0.0
        self.done = False
    def _positions(self):
        sx, sy = self.start_fn(); ex, ey = self.end_fn()
        return (sx, sy), (ex, ey)
    def update(self, dt: float):
        if self.done: return
        (sx, sy), (ex, ey) = self._positions()
        length = max(1e-3, math.hypot(ex - sx, ey - sy))
        self.t += (self.speed * dt) / length
        if self.t >= 1.0: self.t = 1.0; self.done = True
    def pos(self) -> Tuple[float, float]:
        (sx, sy), (ex, ey) = self._positions()
        return (sx + (ex - sx) * self.t, sy + (ey - sy) * self.t)
    def draw(self, surface, dy: int = 0):
        x, y = self.pos(); y += dy
        pygame.draw.circle(surface, (210, 255, 190), (int(x), int(y)), 10)
        t = SMALL.render(self.text, True, (240, 255, 240))
        surface.blit(t, (x + 10, y - t.get_height()/2))

###############################################################################
# Bezier helpers
###############################################################################

def _quad_bezier(p0, p1, p2, t):
    x = (1-t)*(1-t)*p0[0] + 2*(1-t)*t*p1[0] + t*t*p2[0]
    y = (1-t)*(1-t)*p0[1] + 2*(1-t)*t*p1[1] + t*t*p2[1]
    return (x, y)

def _draw_bezier_with_pulse(surface, p0, p2, dy, pulse_phase, alpha=255):
    midx = (p0[0] + p2[0]) * 0.5
    midy = (p0[1] + p2[1]) * 0.5
    ctrl = (midx, midy - 48)

    steps = 28
    pts = []
    for i in range(steps+1):
        t = i/steps
        x, y = _quad_bezier((p0[0], p0[1]+dy), (ctrl[0], ctrl[1]+dy), (p2[0], p2[1]+dy), t)
        pts.append((int(x), int(y)))
    if len(pts) >= 2:
        col = (150,170,230, alpha)
        surf = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        pygame.draw.lines(surf, col, False, pts, 2)
        surface.blit(surf, (0,0))

    tt = (pulse_phase % 1.0)
    px, py = _quad_bezier((p0[0], p0[1]+dy), (ctrl[0], ctrl[1]+dy), (p2[0], p2[1]+dy), tt)
    pygame.draw.circle(surface, (240,245,255), (int(px), int(py)), 6)

###############################################################################
# Panels
###############################################################################

class Machine:
    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.rect = Rect(x, y, 120, 88)
        self.active_timer = 0.0
    def draw(self, surface, dt: float, dy: int = 0):
        if self.active_timer > 0: self.active_timer -= dt
        label = BIG.render(self.name, True, (235, 240, 255))
        padding_x = 28; min_w = 160
        target_w = max(min_w, label.get_width() + padding_x*2)
        cx = self.rect.centerx
        self.rect.w = target_w; self.rect.x = cx - self.rect.w // 2; self.rect.h = 88
        base = (52, 58, 86); glow = (92, 142, 255)
        color = glow if self.active_timer > 0 else base
        draw_rect = Rect(self.rect.x, self.rect.y + dy, self.rect.w, self.rect.h)
        pygame.draw.rect(surface, color, draw_rect, border_radius=18)
        pygame.draw.rect(surface, (180, 200, 255), draw_rect, 2, border_radius=18)
        surface.blit(label, (draw_rect.centerx - label.get_width()/2, draw_rect.y + 8))

class CodePanel:
    def __init__(self, codemap: CodeMap, rect: Rect):
        self.cm = codemap; self.rect = rect
        self.scroll = 0; self.current_lineno: Optional[int] = None
    def set_current_line(self, lineno: Optional[int]):
        self.current_lineno = lineno
        if lineno is None: return
        visible_lines = (self.rect.h - CODE_TOP_PADDING - MARGIN) // LINE_H
        if lineno - 1 < self.scroll:
            self.scroll = max(0, lineno - 1)
        elif lineno - 1 >= self.scroll + visible_lines - 3:
            self.scroll = max(0, lineno - visible_lines + 3)
    def _max_scroll(self) -> int:
        visible_lines = (self.rect.h - CODE_TOP_PADDING - MARGIN) // LINE_H
        return max(0, len(self.cm.lines) - visible_lines)
    def scroll_up(self, n: int = 3): self.scroll = max(0, self.scroll - n)
    def scroll_down(self, n: int = 3): self.scroll = min(self._max_scroll(), self.scroll + n)
    def draw(self, surface):
        pygame.draw.rect(surface, (18,20,28), self.rect, border_radius=12)
        pygame.draw.rect(surface, (80,90,110), self.rect, 1, border_radius=12)
        title = BIG.render("Code", True, (235,235,245))
        surface.blit(title, (self.rect.x + MARGIN, self.rect.y + MARGIN))
        visible_top = self.scroll
        visible_h = (self.rect.h - CODE_TOP_PADDING - MARGIN)
        visible_lines = max(1, visible_h // LINE_H)
        y0 = self.rect.y + CODE_TOP_PADDING
        for (start, end, kind) in self.cm.blocks:
            if end < visible_top + 1 or start > visible_top + visible_lines: continue
            y_start = y0 + (start - 1 - visible_top) * LINE_H
            y_end   = y0 + (end   - 1 - visible_top) * LINE_H + LINE_H
            color = {"if": (40,70,90,60), "for": (70,40,90,60), "while": (60,60,20,60)}.get(kind, (50,50,50,60))
            box = pygame.Surface((self.rect.w - 2*MARGIN, max(8, y_end - y_start)), pygame.SRCALPHA)
            box.fill(color); surface.blit(box, (self.rect.x + MARGIN, y_start))
        for i in range(visible_lines+1):
            idx = visible_top + i
            if idx >= len(self.cm.lines): break
            y = y0 + i * LINE_H; line = self.cm.lines[idx]; num = idx + 1
            surface.blit(SMALL.render(f"{num:4d}", True, (130,130,150)), (self.rect.x + MARGIN, y))
            surface.blit(FONT.render(line.expandtabs(4), True, (230,230,230)), (self.rect.x + MARGIN + 56, y))
        if self.current_lineno is not None and 1 <= self.current_lineno <= len(self.cm.lines):
            y = y0 + (self.current_lineno - 1 - visible_top) * LINE_H
            hi = pygame.Surface((self.rect.w - 2*MARGIN, LINE_H), pygame.SRCALPHA)
            hi.fill((120, 180, 255, 90)); surface.blit(hi, (self.rect.x + MARGIN, y))

class StackPanel:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = Rect(x, y, w, h); self.stack = []; self.scroll_y = 0
        self.title_h = 0; self.pad_box = 10; self.box_gap = 8; self.header_h = 26
        self.line_h = SMALL.get_linesize(); self.value_max_lines = 3
    def set_stack(self, frames): self.stack = frames
    def content_height(self) -> int:
        y = self.title_h; inner_w = self.rect.w - 2*self.pad_box - 12
        for _, locals_preview in self.stack:
            lines_total = 0
            for k, v in list(locals_preview.items()):
                txt = f"{k} = {safe_repr(v, 64)}"
                lines_total += min(self.value_max_lines, len(wrap_text_to_lines(SMALL, txt, inner_w)))
            box_h = self.header_h + 6 + max(self.line_h, lines_total * self.line_h) + 10
            y += box_h + self.box_gap
        return y + self.pad_box
    def scroll_up_pixels(self, px: int = 60): self.scroll_y = max(0, self.scroll_y - px)
    def scroll_down_pixels(self, px: int = 60):
        max_scroll = max(0, self.content_height() - self.rect.h)
        self.scroll_y = min(max_scroll, self.scroll_y + px)
    def draw(self, surface):
        clip_prev = surface.get_clip(); surface.set_clip(self.rect.inflate(-4, -4))
        y_cursor = self.rect.y + self.title_h - self.scroll_y
        inner_x = self.rect.x + self.pad_box; inner_w = self.rect.w - 2*self.pad_box - 12
        for func, locals_preview in self.stack:
            lines_per_var = []; lines_total = 0
            for k, v in list(locals_preview.items()):
                txt = f"{k} = {safe_repr(v, 64)}"
                lines = wrap_text_to_lines(SMALL, txt, inner_w)[:self.value_max_lines]
                lines_per_var.append(lines); lines_total += len(lines)
            box_h = self.header_h + 6 + max(self.line_h, lines_total * self.line_h) + 10
            box = Rect(inner_x, y_cursor, self.rect.w - 2*self.pad_box, box_h)
            if box.bottom < self.rect.y - 2*self.line_h: y_cursor += box_h + self.box_gap; continue
            if box.y > self.rect.bottom: break
            pygame.draw.rect(surface, (46,54,78), box, border_radius=10)
            pygame.draw.rect(surface, (150,170,210), box, 1, border_radius=10)
            surface.blit(FONT.render(func, True, (230,235,245)), (box.x + 8, box.y + 6))
            yy = box.y + self.header_h
            for lines in lines_per_var:
                for ln in lines:
                    surface.blit(SMALL.render(ln, True, (210,215,225)), (box.x + 8, yy))
                    yy += self.line_h
            y_cursor += box_h + self.box_gap
        surface.set_clip(clip_prev)

class VarHistoryPanel:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.rect = Rect(x, y, w, h); self.histories = {}; self.scroll_y = 0
        self.title_h = 0; self.section_gap = 10; self.line_h = SMALL.get_linesize()
    def set_histories(self, histories): self.histories = histories
    def scroll_up_pixels(self, px: int = 60): self.scroll_y = max(0, self.scroll_y - px)
    def _content_height(self) -> int:
        y = self.title_h
        for _, entries in self.histories.items():
            y += 24 + self.line_h * min(8, len(entries)) + self.section_gap
        return y
    def scroll_down_pixels(self, px: int = 60):
        max_scroll = max(0, self._content_height() - self.rect.h)
        self.scroll_y = min(max_scroll, self.scroll_y + px)
    def draw(self, surface):
        clip_prev = surface.get_clip(); surface.set_clip(self.rect.inflate(-4, -4))
        y = self.rect.y + self.title_h - self.scroll_y
        for var in sorted(self.histories.keys()):
            entries = self.histories[var]
            surface.blit(FONT.render(var, True, (230,235,245)), (self.rect.x + 10, y))
            y += 24
            for e in entries[-8:][::-1]:
                line = f"{e['old']} → {e['new']}  @{e['func']}:{e['lineno']}"
                surface.blit(SMALL.render(line, True, (210,215,225)), (self.rect.x + 14, y))
                y += self.line_h
            y += self.section_gap
        surface.set_clip(clip_prev)

###############################################################################
# Edge trail (persist edges with idle TTL)
###############################################################################

class EdgeTrail:
    EDGE_IDLE_TTL = 8.0   # 유지시간 ↑
    FADE_TAIL     = 2.5   # 페이드아웃 더 느리게

    def __init__(self):
        self.map: Dict[Tuple[str, str], Dict[str, float]] = {}

    def touch(self, u: Optional[str], v: Optional[str], weight: float):
        if not u or not v: return
        k = (u, v); now = time.time()
        info = self.map.get(k)
        if info is None:
            self.map[k] = {"last_seen": now, "weight": weight}
        else:
            info["last_seen"] = now
            info["weight"] = max(info["weight"], weight)

    def items_alive(self):
        now = time.time()
        dead_keys = []
        for (u, v), info in self.map.items():
            age = now - info["last_seen"]
            if age <= self.EDGE_IDLE_TTL + self.FADE_TAIL:
                if age <= self.EDGE_IDLE_TTL: alpha = 255
                else:
                    t = min(1.0, (age - self.EDGE_IDLE_TTL) / self.FADE_TAIL)
                    alpha = int(255 * (1.0 - t))
                yield (u, v, age, alpha, info["weight"])
            else:
                dead_keys.append((u, v))
        for k in dead_keys:
            self.map.pop(k, None)

###############################################################################
# Factory panel (쿨링+스냅+충돌해소)
###############################################################################

class FactoryPanel:
    NODE_W_MIN = 160
    NODE_H     = 88
    PADDING    = 60
    K_REPULSE  = 22000
    K_SPRING   = 18.0
    L_SPRING   = 260.0
    DAMPING    = 0.90   # ↑ 안정
    MAX_V      = 450.0  # ↓ 과속 제한
    DT_CLAMP   = 1/60

    GRID_GAP_X = 36
    GRID_GAP_Y = 28

    def __init__(self, rect: Rect):
        self.rect = rect
        self.machines: Dict[str, Machine] = {}
        self.popups: List[Popup] = []
        self.flows: List[DynamicFlowToken] = []
        self.scroll_y = 0

        self.pos: Dict[str, pygame.math.Vector2] = {}
        self.vel: Dict[str, pygame.math.Vector2] = {}

        self.edge_trail = EdgeTrail()

        # 쿨링/중력
        self.cool       = 1.0
        self.cool_decay = 0.992
        self.gravity_k  = 140.0

    def set_rect(self, rect: Rect): self.rect = rect

    def _spawn_position(self) -> pygame.math.Vector2:
        cx = self.rect.centerx; cy = self.rect.centery
        r  = min(self.rect.w, self.rect.h) * 0.28
        i  = len(self.pos)
        ang = math.tau * ((i * 0.318) % 1.0)  # 골든앵글 분산
        x = cx + r * math.cos(ang) + ((i*37) % 17 - 8)
        y = cy + r * math.sin(ang) + ((i*53) % 19 - 9)
        return pygame.math.Vector2(x, y)

    def _ensure_func(self, func: str):
        if func in self.machines: return
        m = Machine(func, self.rect.centerx-60, self.rect.y + self.PADDING)
        self.machines[func] = m
        self.pos[func] = self._spawn_position()
        self.vel[func] = pygame.math.Vector2(0, 0)
        self.cool = 1.0  # 새 노드 등장 시 다시 살짝 흔들기

    def anchor_left(self, func: str) -> Tuple[int, int]:
        m = self.machines[func]; return (m.rect.left, m.rect.centery)
    def anchor_right(self, func: str) -> Tuple[int, int]:
        m = self.machines[func]; return (m.rect.right, m.rect.centery)

    def popup(self, text: str, func: str, left=True, ttl: float = POPUP_TTL):
        self._ensure_func(func)
        m = self.machines[func]
        x = m.rect.x - 170 if left else m.rect.right + 16
        self.popups.append(Popup(text, x, m.rect.y - 10, ttl))

    def add_call(self, caller: Optional[str], callee: str, args_text: str):
        self._ensure_func(callee)
        if caller:
            self._ensure_func(caller)
            start_fn = lambda c=caller: self.anchor_right(c)
            end_fn   = lambda d=callee: self.anchor_left(d)
            self.flows.append(DynamicFlowToken(f"args({args_text})", start_fn, end_fn, speed=260.0))
            self.edge_trail.touch(caller, callee, 1.0)
        self.machines[callee].active_timer = 0.3
        self.popup(f"call {callee}()", callee, left=True)

    def add_return(self, caller: Optional[str], callee: str, ret_text: str):
        self._ensure_func(callee)
        if caller:
            self._ensure_func(caller)
            start_fn = lambda d=callee: self.anchor_right(d)
            end_fn   = lambda c=caller: self.anchor_left(c)
            self.flows.append(DynamicFlowToken(f"ret({ret_text})", start_fn, end_fn, speed=220.0))
            self.edge_trail.touch(callee, caller, 0.7)
        self.machines[callee].active_timer = 0.3
        self.popup(f"return {ret_text}", callee, left=False)

    def content_height(self) -> int:
        return max(self.rect.h, int(self.rect.h * 1.2))
    def scroll_up(self, pixels: int = 60): self.scroll_y = max(0, self.scroll_y - pixels)
    def scroll_down(self, pixels: int = 60):
        max_scroll = max(0, self.content_height() - self.rect.h)
        self.scroll_y = min(max_scroll, self.scroll_y + pixels)

    def _physics_step(self, dt: float):
        if not self.machines: return
        dt = min(dt, self.DT_CLAMP)
        nodes = list(self.machines.keys())

        # 1) 반발력
        for i in range(len(nodes)):
            ni = nodes[i]
            for j in range(i+1, len(nodes)):
                nj = nodes[j]
                d = self.pos[nj] - self.pos[ni]
                dist2 = max(1.0, d.length_squared())
                if dist2 > 1.0:
                    f = d.normalize() * (self.K_REPULSE / dist2)
                    self.vel[ni] -= f * dt
                    self.vel[nj] += f * dt

        # 2) 스프링(유지 간선 기반, 나이 들수록 약화)
        for (u, v, age, _, w) in self.edge_trail.items_alive():
            if u not in self.pos or v not in self.pos: continue
            d = self.pos[v] - self.pos[u]
            L = d.length() or 1.0
            dir = d / L
            k = self.K_SPRING * (0.9 ** age) * (0.6 + 0.4*w)
            F = k * (L - self.L_SPRING) * dir
            self.vel[u] += F * dt
            self.vel[v] -= F * dt

        # 3) 경계 + 마찰
        left   = self.rect.x + self.PADDING
        right  = self.rect.right - self.PADDING
        top    = self.rect.y + self.PADDING
        bottom = self.rect.bottom - self.PADDING

        # 3.5) 중앙 중력(쿨링 비례)
        center = pygame.math.Vector2(self.rect.centerx, self.rect.centery)

        for n in nodes:
            to_c = (center - self.pos[n])
            self.vel[n] += to_c * (self.gravity_k * self.cool) * dt / max(200.0, to_c.length() or 1)

            self.vel[n] *= self.DAMPING
            if self.vel[n].length() > self.MAX_V:
                self.vel[n].scale_to_length(self.MAX_V)

            self.pos[n] += self.vel[n] * dt
            p = self.pos[n]; bounced = False
            if p.x < left:   p.x = left;   self.vel[n].x *= -0.4; bounced = True
            elif p.x > right:p.x = right;  self.vel[n].x *= -0.4; bounced = True
            if p.y < top:    p.y = top;    self.vel[n].y *= -0.4; bounced = True
            elif p.y > bottom:p.y = bottom;self.vel[n].y *= -0.4; bounced = True
            if bounced: self.vel[n] *= 0.7

        # 4) 충돌 해소(겹침 분리)
        def rect_of(name: str) -> Rect:
            m = self.machines[name]
            w = max(self.NODE_W_MIN, m.rect.w)
            return Rect(int(self.pos[name].x - w/2), int(self.pos[name].y - self.NODE_H/2), int(w), self.NODE_H)

        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a = nodes[i]; b = nodes[j]
                ra = rect_of(a); rb = rect_of(b)
                if ra.colliderect(rb):
                    dx1 = rb.right - ra.left
                    dx2 = ra.right - rb.left
                    dy1 = rb.bottom - ra.top
                    dy2 = ra.bottom - rb.top
                    move_x = dx1 if dx1 < dx2 else -dx2
                    move_y = dy1 if dy1 < dy2 else -dy2
                    if abs(move_x) < abs(move_y):
                        sep = pygame.math.Vector2(move_x/2, 0)
                    else:
                        sep = pygame.math.Vector2(0, move_y/2)
                    self.pos[a] -= sep
                    self.pos[b] += sep

        # 4.5) Soft grid snap – 정렬감 부여
        cell_w = self.NODE_W_MIN + self.GRID_GAP_X
        cell_h = self.NODE_H     + self.GRID_GAP_Y
        grid_cols = max(1, int((self.rect.w - 2*self.PADDING) // cell_w))
        grid_rows = max(1, int((self.rect.h - 2*self.PADDING) // cell_h))
        gx0 = self.rect.centerx - (grid_cols-1)*cell_w/2
        gy0 = self.rect.centery - (grid_rows-1)*cell_h/2
        occupied = set()
        for name in nodes:
            px, py = self.pos[name].x, self.pos[name].y
            cx = int(round((px - gx0) / cell_w))
            cy = int(round((py - gy0) / cell_h))
            cx = max(0, min(grid_cols-1, cx))
            cy = max(0, min(grid_rows-1, cy))
            found = (cx, cy)
            if (cx, cy) in occupied:
                best = None; best_d2 = 1e18
                for r in range(1, 3):
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            nx, ny = cx+dx, cy+dy
                            if not (0<=nx<grid_cols and 0<=ny<grid_rows): continue
                            if (nx, ny) in occupied: continue
                            sx = gx0 + nx*cell_w; sy = gy0 + ny*cell_h
                            d2 = (sx-px)*(sx-px) + (sy-py)*(sy-py)
                            if d2 < best_d2: best_d2 = d2; best = (nx, ny)
                if best is not None: found = best
            occupied.add(found)
            slot_x = gx0 + found[0]*cell_w
            slot_y = gy0 + found[1]*cell_h
            target = pygame.math.Vector2(slot_x, slot_y)
            alpha = 0.10 + 0.25 * self.cool   # 쿨링 비례 보간
            self.pos[name] = self.pos[name].lerp(target, alpha)

        # 5) 머신 rect 동기화
        for name, m in self.machines.items():
            cx, cy = int(self.pos[name].x), int(self.pos[name].y)
            m.rect.centerx = cx; m.rect.centery = cy

        # 6) 쿨링 감소 및 고정
        self.cool *= self.cool_decay
        if self.cool < 0.12:
            for n in nodes:
                self.vel[n] *= 0.15

    def update(self, dt: float):
        self._physics_step(dt)
        for p in list(self.popups):
            p.update(dt)
            if p.ttl <= 0: self.popups.remove(p)
        for f in list(self.flows):
            f.update(dt)
            if f.done: self.flows.remove(f)

    def draw(self, surface):
        pygame.draw.rect(surface, (24,26,34), self.rect, border_radius=12)
        pygame.draw.rect(surface, (80,90,110), self.rect, 1, border_radius=12)
        title = BIG.render("Factory", True, (235,235,245))
        surface.blit(title, (self.rect.x + 12, self.rect.y + 10))

        dy = -self.scroll_y

        # 유지 간선
        phase_base = (time.time() * 0.8) % 1.0
        for (u, v, age, alpha, w) in self.edge_trail.items_alive():
            if u not in self.machines or v not in self.machines: continue
            su = (self.machines[u].rect.centerx, self.machines[u].rect.centery)
            ev = (self.machines[v].rect.centerx, self.machines[v].rect.centery)
            _draw_bezier_with_pulse(surface, su, ev, dy, pulse_phase=(phase_base * (0.7 + 0.3*w)), alpha=alpha)

        # 진행 중 화살표/토큰
        for flow in self.flows:
            s = flow.start_fn(); e = flow.end_fn()
            draw_arrow(surface, (s[0], s[1]+dy), (e[0], e[1]+dy), color=(170, 200, 255), width=2, head_len=12)
        for flow in self.flows:
            flow.draw(surface, dy)

        for m in self.machines.values():
            m.draw(surface, 0, dy)
        for p in self.popups:
            p.draw(surface, dy)

###############################################################################
# Tabbed sidebar
###############################################################################

class TabbedSidebar:
    def __init__(self, x: int, y: int, w: int, h: int, stack_panel: "StackPanel", var_panel: "VarHistoryPanel"):
        self.rect = Rect(x, y, w, h)
        self.stack_panel = stack_panel; self.var_panel = var_panel
        self.active = "stack"; self.tab_h = 36; self.tab_gap = 8
    def set_geometry(self, x: int, y: int, w: int, h: int): self.rect.update(x, y, w, h)
    def handle_mouse(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self.rect.collidepoint(mx, my):
                left_tab = Rect(self.rect.x + 10, self.rect.y + 6, 80, self.tab_h-12)
                right_tab = Rect(left_tab.right + self.tab_gap, self.rect.y + 6, 80, self.tab_h-12)
                if left_tab.collidepoint(mx, my): self.active = "stack"
                elif right_tab.collidepoint(mx, my): self.active = "var"
    def scroll(self, up: bool):
        if self.active == "stack": (self.stack_panel.scroll_up_pixels if up else self.stack_panel.scroll_down_pixels)(80)
        else: (self.var_panel.scroll_up_pixels if up else self.var_panel.scroll_down_pixels)(80)
    def draw(self, surface):
        pygame.draw.rect(surface, (24,26,34), self.rect, border_radius=12)
        pygame.draw.rect(surface, (80,90,110), self.rect, 1, border_radius=12)
        left_tab = Rect(self.rect.x + 10, self.rect.y + 6, 80, self.tab_h-12)
        right_tab = Rect(left_tab.right + self.tab_gap, self.rect.y + 6, 80, self.tab_h-12)
        def draw_tab(r: Rect, label: str, active: bool):
            col = (70,90,120) if active else (46,54,78)
            pygame.draw.rect(surface, col, r, border_radius=8)
            pygame.draw.rect(surface, (150,170,210), r, 1, border_radius=8)
            t = FONT.render(label, True, (235,235,245))
            surface.blit(t, (r.centerx - t.get_width()//2, r.centery - t.get_height()//2))
        draw_tab(left_tab,  "Stack", self.active=="stack")
        draw_tab(right_tab, "Var",   self.active=="var")
        inner = Rect(self.rect.x+8, self.rect.y + self.tab_h, self.rect.w-16, self.rect.h - self.tab_h - 8)
        if self.active == "stack":
            self.stack_panel.rect = inner.copy(); self.stack_panel.draw(surface)
        else:
            self.var_panel.rect = inner.copy(); self.var_panel.draw(surface)

###############################################################################
# Main app
###############################################################################

class VisualDebuggerApp:
    def __init__(self, codemap: CodeMap, event_queue: "queue.Queue[dict]", file_filter: str):
        self.events: Deque[dict] = deque(); self.q = event_queue; self.file_filter = file_filter
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Factory Visual Debugger — Calm Layout (Cooling+Snap+Trail)")
        self.clock = pygame.time.Clock()
        self.codemap = codemap
        self._compute_layout()
        self.code_panel   = CodePanel(self.codemap, self.rect_code)
        self.factory      = FactoryPanel(self.rect_factory)
        self.stack_panel  = StackPanel(0,0,0,0)
        self.var_panel    = VarHistoryPanel(0,0,0,0)
        self.sidebar_tabs = TabbedSidebar(self.rect_sidebar.x, self.rect_sidebar.y, self.rect_sidebar.w, self.rect_sidebar.h,
                                          self.stack_panel, self.var_panel)
        self.started = False; self.step_hold = 0.0
        self.stack_frames: List[Tuple[str, Dict[str, Any]]] = []
        self.var_history: Dict[str, List[Dict[str, Any]]] = {}

    def _compute_layout(self):
        top_h = int(HEIGHT * TOP_FACTORY_RATIO)
        self.rect_factory = Rect(MARGIN, MARGIN, WIDTH - 2*MARGIN, top_h - MARGIN)
        bottom_y = self.rect_factory.bottom + MARGIN
        bottom_h = HEIGHT - bottom_y - MARGIN
        sidebar_w = max(RIGHT_SIDEBAR_MIN_W, int(WIDTH * (1 - BOTTOM_CODE_RATIO)))
        code_w = WIDTH - 3*MARGIN - sidebar_w
        self.rect_code    = Rect(MARGIN, bottom_y, code_w, bottom_h)
        self.rect_sidebar = Rect(self.rect_code.right + MARGIN, bottom_y, sidebar_w, bottom_h)

    def push_stack(self, func: str, args: Dict[str, Any]):
        preview = dict(list(args.items())[:3]); self.stack_frames.insert(0, (func, preview))
    def pop_stack(self, func: str):
        for i, (fname, _) in enumerate(self.stack_frames):
            if fname == func: del self.stack_frames[i]; break
    def set_line(self, lineno: Optional[int]): self.code_panel.set_current_line(lineno)

    def handle_event(self, ev: dict):
        et = ev["type"]
        if not self.started:
            self.started = True
            self.factory.popups.append(Popup("Input received — stepping…", self.rect_factory.x + 20, self.rect_factory.y + 16, ttl=3.0))
        if et == "call":
            func = ev["func"]; caller = ev.get("caller")
            args = ev.get("args", {})
            arg_text = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
            if len(args) > 3: arg_text += ", ..."
            self.push_stack(func, args)
            self.factory.add_call(caller, func, arg_text)
            if isinstance(ev.get("lineno"), int): self.set_line(ev["lineno"])
        elif et == "return":
            func = ev["func"]; caller = ev.get("caller"); val = safe_repr(ev.get("value"))
            self.pop_stack(func); self.factory.add_return(caller, func, val)
        elif et == "assign":
            func = ev["func"]; var = ev["var"]
            old_raw = ev.get("old"); new_raw = ev.get("new", ev.get("value"))
            old = "—" if old_raw is None else str(old_raw)
            new = "—" if new_raw is None else str(new_raw)
            self.factory.popup(f"{var}: {old} → {new}", func, left=False)
            hist = self.var_history.setdefault(var, [])
            hist.append({"old": old, "new": new, "func": func, "lineno": ev.get("lineno", 0)})
            self.var_panel.set_histories(self.var_history)
        elif et == "line":
            if isinstance(ev.get("lineno"), int): self.set_line(ev["lineno"])
        self.step_hold = STEP_PAUSE

    def _handle_mouse_wheel(self, event, mx, my):
        wheel_up = (event.button == 4)
        if self.rect_factory.collidepoint(mx, my):
            (self.factory.scroll_up if wheel_up else self.factory.scroll_down)(80)
        elif self.rect_code.collidepoint(mx, my):
            (self.code_panel.scroll_up if wheel_up else self.code_panel.scroll_down)(3)
        elif self.rect_sidebar.collidepoint(mx, my):
            self.sidebar_tabs.scroll(up=wheel_up)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button in (4, 5): self._handle_mouse_wheel(event, mx, my)
                    else: self.sidebar_tabs.handle_mouse(event)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: self.code_panel.scroll_up(1)
                    elif event.key == pygame.K_DOWN: self.code_panel.scroll_down(1)
                    elif event.key == pygame.K_PAGEUP: self.code_panel.scroll_up(10)
                    elif event.key == pygame.K_PAGEDOWN: self.code_panel.scroll_down(10)
                    elif event.key == pygame.K_w and pygame.key.get_mods() & pygame.KMOD_SHIFT: self.factory.scroll_up(80)
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_SHIFT: self.factory.scroll_down(80)
                    elif event.key == pygame.K_w and pygame.key.get_mods() & pygame.KMOD_ALT: self.sidebar_tabs.scroll(True)
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_ALT: self.sidebar_tabs.scroll(False)
                    elif event.key == pygame.K_TAB:
                        self.sidebar_tabs.active = "var" if self.sidebar_tabs.active == "stack" else "stack"

            try:
                while True:
                    ev = self.q.get_nowait()
                    self.events.append(ev)
            except queue.Empty:
                pass

            self.factory.update(dt)
            self.stack_panel.set_stack(self.stack_frames)
            self.var_panel.set_histories(self.var_history)

            if self.step_hold > 0: self.step_hold -= dt
            else:
                if not self.factory.flows and self.events:
                    ev = self.events.popleft()
                    self.handle_event(ev)

            self.screen.fill((18,20,28))
            self.factory.draw(self.screen)
            self.code_panel.draw(self.screen)
            self.sidebar_tabs.draw(self.screen)
            pygame.display.flip()
        pygame.quit()

###############################################################################
# Runner
###############################################################################

ENTRY_CANDIDATES = ["run_target", "main", "run"]

def traced_runner(q: "queue.Queue[dict]", code_src: str, filename: str):
    tc = TraceCollector(q, filename)
    compiled = compile(code_src, filename, "exec")
    ns: Dict[str, Any] = {"__name__": "__main__"}
    sys.settrace(tc.tracer)
    try:
        exec(compiled, ns, ns)
        entry = None
        for name in ENTRY_CANDIDATES:
            fn = ns.get(name)
            if callable(fn):
                entry = fn; break
        if entry is not None:
            result = entry()
            q.put({"type": "assign", "func": entry.__name__, "var": "RESULT", "value": safe_repr(result), "ts": time.time()})
    finally:
        sys.settrace(None)

###############################################################################
# Bootstrap
###############################################################################

if __name__ == "__main__":
    codemap = CodeMap(source=TEST_CODE, pseudo_name=TEST_FILENAME)
    evq: "queue.Queue[dict]" = queue.Queue()
    t = threading.Thread(target=traced_runner, args=(evq, TEST_CODE, TEST_FILENAME), daemon=True)
    t.start()
    app = VisualDebuggerApp(codemap, evq, TEST_FILENAME)
    app.run()
