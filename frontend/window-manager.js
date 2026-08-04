/**
 * WindowManager — generic draggable/resizable/closable node panel system.
 *
 * Domain-agnostic: no Vocalize-specific knowledge.  Manages absolute-positioned
 * nodes on a canvas element with grid snapping, edge snapping, z-order,
 * localStorage persistence, and an Add Node visibility toggle panel.
 *
 * Usage:
 *   const wm = new WindowManager(canvasEl, { storageKey, gridSize, snapThreshold });
 *   wm.register(id, { title, contentEl, footerEl, defaultRect, minSize, visible });
 *   wm.mount();        // render all nodes into the canvas
 *   wm.resetLayout();  // restore defaults + clear localStorage
 */

class WindowManager {
    /**
     * @param {HTMLElement} canvas   — the container element (position: fixed/absolute)
     * @param {Object}      opts
     * @param {string}      opts.storageKey     — localStorage key for layout persistence
     * @param {number}      opts.gridSize       — snap grid unit in px (default 24)
     * @param {number}      opts.snapThreshold  — edge-snap proximity in px (default 8)
     */
    constructor(canvas, opts = {}) {
        this._canvas = canvas;
        this._storageKey = opts.storageKey || 'wm-layout';
        this._gridSize = opts.gridSize || 24;
        this._snapThresh = opts.snapThreshold || 8;

        // Read toolbar height from DOM so we never hardcode it
        const toolbar = document.querySelector('.toolbar');
        this._toolbarH = toolbar ? toolbar.getBoundingClientRect().height : 48;

        this._nodes = new Map();          // id → NodeState
        this._zCounter = 100;             // shared z-index counter
        this._dragState = null;           // active drag pointer state
        this._resizeState = null;         // active resize pointer state
        this._guides = [];                // active snap guide DOM elements
        this._saveTimer = null;           // debounce timer for persistence
        this._onVisibilityChange = null;  // callback: (id, visible) => {}
    }

    // ────────────────────────────────────
    // Registration
    // ────────────────────────────────────

    /**
     * Register a node definition (does NOT create DOM yet — call mount() after).
     *
     * @param {string} id             — unique identifier
     * @param {Object} def
     * @param {string} def.title      — displayed in the header (e.g. "CH.01 · WEBCAM")
     * @param {HTMLElement} def.contentEl  — the existing DOM element with live content
     * @param {HTMLElement} [def.footerEl] — optional sticky footer element
     * @param {{x,y,w,h}} def.defaultRect — default position/size (px, pre-snapped to grid)
     * @param {{w,h}}     def.minSize     — minimum width/height
     * @param {boolean}   def.visible     — whether visible by default
     */
    register(id, def) {
        this._nodes.set(id, {
            id,
            title: def.title,
            contentEl: def.contentEl,
            footerEl: def.footerEl || null,
            defaultRect: { ...def.defaultRect },
            minSize: { w: def.minSize?.w || 260, h: def.minSize?.h || 160 },
            rect: { ...def.defaultRect },
            visible: def.visible !== false,
            z: this._zCounter++,
            el: null,         // will be set by _createNodeDOM
            headerEl: null,
            chDot: null,
        });
    }

    // ────────────────────────────────────
    // Mount: build DOM + apply saved layout
    // ────────────────────────────────────

    mount() {
        // Load saved layout from localStorage (if any)
        const saved = this._load();

        this._nodes.forEach((node, id) => {
            // Apply saved rect/visibility if available
            if (saved && saved[id]) {
                node.rect = { ...saved[id].rect };
                node.visible = saved[id].visible;
                node.z = saved[id].z || node.z;
            }

            this._createNodeDOM(node);
            this._applyRect(node);
            node.el.style.zIndex = node.z;

            if (!node.visible) {
                node.el.style.display = 'none';
            }

            this._canvas.appendChild(node.el);
        });
    }

    // ────────────────────────────────────
    // DOM construction
    // ────────────────────────────────────

    _createNodeDOM(node) {
        const el = document.createElement('div');
        el.className = 'wm-node';
        el.dataset.wmId = node.id;

        // ── Header ──
        const header = document.createElement('div');
        header.className = 'wm-node-header';

        const dot = document.createElement('span');
        dot.className = 'ch-dot';
        node.chDot = dot;

        const tag = document.createElement('span');
        tag.className = 'ch-tag';
        tag.textContent = node.title;

        const spacer = document.createElement('span');
        spacer.className = 'wm-header-spacer';

        const closeBtn = document.createElement('button');
        closeBtn.className = 'wm-close';
        closeBtn.textContent = '\u00d7';
        closeBtn.addEventListener('pointerdown', (e) => {
            e.stopPropagation();
        });
        closeBtn.addEventListener('click', () => {
            this.hide(node.id);
        });

        header.appendChild(dot);
        header.appendChild(tag);
        header.appendChild(spacer);
        header.appendChild(closeBtn);
        node.headerEl = header;

        // ── Body ──
        const body = document.createElement('div');
        body.className = 'wm-node-body';
        if (node.contentEl) {
            body.appendChild(node.contentEl);
            node.contentEl.style.display = '';
        }

        el.appendChild(header);
        el.appendChild(body);

        // ── Footer (optional) ──
        if (node.footerEl) {
            const footer = document.createElement('div');
            footer.className = 'wm-node-footer';
            footer.appendChild(node.footerEl);
            node.footerEl.style.display = '';
            el.appendChild(footer);
        }

        // ── Resize handles ──
        const dirs = ['n', 's', 'w', 'e', 'nw', 'ne', 'sw', 'se'];
        dirs.forEach(dir => {
            const handle = document.createElement('div');
            handle.className = `wm-resize wm-resize-${dir}`;
            handle.dataset.dir = dir;
            handle.addEventListener('pointerdown', (e) => this._onResizeStart(e, node, dir));
            el.appendChild(handle);
        });

        // ── Focus on pointer down ──
        el.addEventListener('pointerdown', () => this._focus(node));

        // ── Drag ──
        header.addEventListener('pointerdown', (e) => this._onDragStart(e, node));

        node.el = el;
    }

    // ────────────────────────────────────
    // Rect helpers
    // ────────────────────────────────────

    _applyRect(node) {
        const el = node.el;
        el.style.left = node.rect.x + 'px';
        el.style.top = node.rect.y + 'px';
        el.style.width = node.rect.w + 'px';
        el.style.height = node.rect.h + 'px';
    }

    _snapToGrid(val) {
        return Math.round(val / this._gridSize) * this._gridSize;
    }

    _clampRect(node) {
        const r = node.rect;
        const min = node.minSize;
        r.w = Math.max(r.w, min.w);
        r.h = Math.max(r.h, min.h);
        // Keep within canvas bounds
        const cw = this._canvas.clientWidth;
        const ch = this._canvas.clientHeight;
        r.x = Math.max(0, Math.min(r.x, cw - 60));
        r.y = Math.max(0, Math.min(r.y, ch - 40));
    }

    // ────────────────────────────────────
    // Focus (z-order)
    // ────────────────────────────────────

    _focus(node) {
        // Remove focused class from all nodes
        this._nodes.forEach(n => {
            if (n.el) n.el.classList.remove('wm-focused');
        });
        node.z = ++this._zCounter;
        node.el.style.zIndex = node.z;
        node.el.classList.add('wm-focused');
    }

    // ────────────────────────────────────
    // Drag
    // ────────────────────────────────────

    _onDragStart(e, node) {
        // Don't drag if clicking close button
        if (e.target.closest('.wm-close')) return;

        e.preventDefault();
        this._focus(node);

        const rect = node.rect;
        this._dragState = {
            node,
            startX: e.clientX,
            startY: e.clientY,
            origX: rect.x,
            origY: rect.y,
            pointerId: e.pointerId,
        };

        node.headerEl.setPointerCapture(e.pointerId);

        const onMove = (ev) => this._onDragMove(ev);
        const onUp = (ev) => {
            this._onDragEnd(ev);
            node.headerEl.removeEventListener('pointermove', onMove);
            node.headerEl.removeEventListener('pointerup', onUp);
            node.headerEl.removeEventListener('pointercancel', onUp);
        };

        node.headerEl.addEventListener('pointermove', onMove);
        node.headerEl.addEventListener('pointerup', onUp);
        node.headerEl.addEventListener('pointercancel', onUp);
    }

    _onDragMove(e) {
        const ds = this._dragState;
        if (!ds) return;

        const dx = e.clientX - ds.startX;
        const dy = e.clientY - ds.startY;
        let newX = ds.origX + dx;
        let newY = ds.origY + dy;

        // Edge snapping against other visible nodes
        this._clearGuides();
        const moving = ds.node;
        const movingEdges = {
            left: newX,
            right: newX + moving.rect.w,
            top: newY,
            bottom: newY + moving.rect.h,
        };

        // Canvas edges
        const canvasEdgesX = [0, this._canvas.clientWidth];
        const canvasEdgesY = [0, this._canvas.clientHeight];

        let snappedX = false, snappedY = false;

        this._nodes.forEach((other) => {
            if (other.id === moving.id || !other.visible) return;
            const or = other.rect;
            const otherEdges = {
                left: or.x,
                right: or.x + or.w,
                top: or.y,
                bottom: or.y + or.h,
            };

            // X-axis snaps
            if (!snappedX) {
                const xPairs = [
                    [movingEdges.left, otherEdges.left],
                    [movingEdges.left, otherEdges.right],
                    [movingEdges.right, otherEdges.left],
                    [movingEdges.right, otherEdges.right],
                ];
                for (const [mEdge, oEdge] of xPairs) {
                    if (Math.abs(mEdge - oEdge) < this._snapThresh) {
                        const shift = oEdge - mEdge;
                        newX += shift;
                        snappedX = true;
                        this._showGuide('v', oEdge);
                        break;
                    }
                }
            }

            // Y-axis snaps
            if (!snappedY) {
                const yPairs = [
                    [movingEdges.top, otherEdges.top],
                    [movingEdges.top, otherEdges.bottom],
                    [movingEdges.bottom, otherEdges.top],
                    [movingEdges.bottom, otherEdges.bottom],
                ];
                for (const [mEdge, oEdge] of yPairs) {
                    if (Math.abs(mEdge - oEdge) < this._snapThresh) {
                        const shift = oEdge - mEdge;
                        newY += shift;
                        snappedY = true;
                        this._showGuide('h', oEdge);
                        break;
                    }
                }
            }
        });

        // Canvas boundary snaps
        if (!snappedX) {
            for (const cx of canvasEdgesX) {
                if (Math.abs(newX - cx) < this._snapThresh) {
                    newX = cx; snappedX = true; this._showGuide('v', cx); break;
                }
                if (Math.abs((newX + moving.rect.w) - cx) < this._snapThresh) {
                    newX = cx - moving.rect.w; snappedX = true; this._showGuide('v', cx); break;
                }
            }
        }
        if (!snappedY) {
            for (const cy of canvasEdgesY) {
                if (Math.abs(newY - cy) < this._snapThresh) {
                    newY = cy; snappedY = true; this._showGuide('h', cy); break;
                }
                if (Math.abs((newY + moving.rect.h) - cy) < this._snapThresh) {
                    newY = cy - moving.rect.h; snappedY = true; this._showGuide('h', cy); break;
                }
            }
        }

        moving.rect.x = newX;
        moving.rect.y = newY;
        this._applyRect(moving);
    }

    _onDragEnd(_e) {
        const ds = this._dragState;
        if (!ds) return;

        this._clearGuides();

        // Grid snap on release
        const node = ds.node;
        node.rect.x = this._snapToGrid(node.rect.x);
        node.rect.y = this._snapToGrid(node.rect.y);
        this._clampRect(node);
        this._applyRect(node);

        this._dragState = null;
        this._scheduleSave();
    }

    // ────────────────────────────────────
    // Resize
    // ────────────────────────────────────

    _onResizeStart(e, node, dir) {
        e.preventDefault();
        e.stopPropagation();
        this._focus(node);

        const rect = { ...node.rect };
        this._resizeState = {
            node,
            dir,
            startX: e.clientX,
            startY: e.clientY,
            origRect: rect,
            pointerId: e.pointerId,
        };

        const handle = e.target;
        handle.setPointerCapture(e.pointerId);

        const onMove = (ev) => this._onResizeMove(ev);
        const onUp = (ev) => {
            this._onResizeEnd(ev);
            handle.removeEventListener('pointermove', onMove);
            handle.removeEventListener('pointerup', onUp);
            handle.removeEventListener('pointercancel', onUp);
        };

        handle.addEventListener('pointermove', onMove);
        handle.addEventListener('pointerup', onUp);
        handle.addEventListener('pointercancel', onUp);
    }

    _onResizeMove(e) {
        const rs = this._resizeState;
        if (!rs) return;

        const dx = e.clientX - rs.startX;
        const dy = e.clientY - rs.startY;
        const or = rs.origRect;
        const node = rs.node;
        const min = node.minSize;
        const r = node.rect;

        if (rs.dir.includes('e')) {
            r.w = Math.max(min.w, or.w + dx);
        }
        if (rs.dir.includes('w')) {
            const newW = Math.max(min.w, or.w - dx);
            r.x = or.x + (or.w - newW);
            r.w = newW;
        }
        if (rs.dir.includes('s')) {
            r.h = Math.max(min.h, or.h + dy);
        }
        if (rs.dir.includes('n')) {
            const newH = Math.max(min.h, or.h - dy);
            r.y = or.y + (or.h - newH);
            r.h = newH;
        }

        this._applyRect(node);
    }

    _onResizeEnd(_e) {
        const rs = this._resizeState;
        if (!rs) return;

        const node = rs.node;
        node.rect.x = this._snapToGrid(node.rect.x);
        node.rect.y = this._snapToGrid(node.rect.y);
        node.rect.w = this._snapToGrid(node.rect.w);
        node.rect.h = this._snapToGrid(node.rect.h);
        this._clampRect(node);
        this._applyRect(node);

        this._resizeState = null;
        this._scheduleSave();
    }

    // ────────────────────────────────────
    // Snap guides
    // ────────────────────────────────────

    _showGuide(axis, pos) {
        const guide = document.createElement('div');
        guide.className = axis === 'h' ? 'snap-guide snap-guide-h' : 'snap-guide snap-guide-v';
        if (axis === 'h') {
            guide.style.top = pos + 'px';
        } else {
            guide.style.left = pos + 'px';
        }
        this._canvas.appendChild(guide);
        this._guides.push(guide);
    }

    _clearGuides() {
        this._guides.forEach(g => g.remove());
        this._guides = [];
    }

    // ────────────────────────────────────
    // Show / Hide
    // ────────────────────────────────────

    show(id) {
        const node = this._nodes.get(id);
        if (!node) return;
        node.visible = true;
        node.el.style.display = '';
        this._focus(node);
        this._scheduleSave();
        if (this._onVisibilityChange) this._onVisibilityChange(id, true);
    }

    hide(id) {
        const node = this._nodes.get(id);
        if (!node) return;
        node.visible = false;
        node.el.style.display = 'none';
        this._scheduleSave();
        if (this._onVisibilityChange) this._onVisibilityChange(id, false);
    }

    isVisible(id) {
        const node = this._nodes.get(id);
        return node ? node.visible : false;
    }

    toggle(id) {
        if (this.isVisible(id)) this.hide(id);
        else this.show(id);
    }

    /** Set a callback for visibility changes (used to update Add Node panel toggles) */
    onVisibilityChange(cb) {
        this._onVisibilityChange = cb;
    }

    /** Get the channel-dot element for a node (to toggle .active class) */
    getChDot(id) {
        const node = this._nodes.get(id);
        return node ? node.chDot : null;
    }

    /** Insert an extra element into a node's header (e.g. count badges) */
    addToHeader(id, el) {
        const node = this._nodes.get(id);
        if (!node || !node.headerEl) return;
        // Insert before the spacer
        const spacer = node.headerEl.querySelector('.wm-header-spacer');
        if (spacer) {
            node.headerEl.insertBefore(el, spacer);
        } else {
            node.headerEl.appendChild(el);
        }
    }

    // ────────────────────────────────────
    // Layout reset
    // ────────────────────────────────────

    resetLayout() {
        localStorage.removeItem(this._storageKey);
        this._nodes.forEach(node => {
            node.rect = { ...node.defaultRect };
            node.visible = node.defaultRect._defaultVisible !== false;
            node.z = 100;
            this._applyRect(node);
            node.el.style.zIndex = node.z;
            node.el.style.display = node.visible ? '' : 'none';
        });
        this._zCounter = 100 + this._nodes.size;
        if (this._onVisibilityChange) {
            this._nodes.forEach(node => {
                this._onVisibilityChange(node.id, node.visible);
            });
        }
    }

    // ────────────────────────────────────
    // Persistence (localStorage)
    // ────────────────────────────────────

    _load() {
        try {
            const raw = localStorage.getItem(this._storageKey);
            if (!raw) return null;
            return JSON.parse(raw);
        } catch {
            return null;
        }
    }

    _save() {
        const data = {};
        this._nodes.forEach((node, id) => {
            data[id] = {
                rect: { ...node.rect },
                visible: node.visible,
                z: node.z,
            };
        });
        try {
            localStorage.setItem(this._storageKey, JSON.stringify(data));
        } catch (e) {
            console.warn('[WM] Save failed:', e);
        }
    }

    _scheduleSave() {
        if (this._saveTimer) clearTimeout(this._saveTimer);
        this._saveTimer = setTimeout(() => this._save(), 300);
    }
}
