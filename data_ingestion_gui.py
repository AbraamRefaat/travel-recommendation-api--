"""
data_ingestion_gui.py
---------------------
A Tkinter desktop application that lets users populate the Qdrant vector
database used by the travel recommendation API.

Two modes:
  1. Manual Entry  — fill a form row-by-row, preview, then click Proceed.
  2. Excel Upload  — pick an .xlsx / .xls file, preview it, then click Proceed.

After clicking Proceed the app:
  • Encodes each row into a sentence
  • Embeds it with SentenceTransformer ("all-MiniLM-L6-v2")
  • Upserts the resulting vector + payload into the Qdrant collection
    (creates the collection automatically if it doesn't exist)

Requirements: pip install sentence-transformers qdrant-client pandas openpyxl
Qdrant must be running (Docker): docker run -d -p 6333:6333 qdrant/qdrant
"""

import os
import sys
import uuid
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------------------
# Load .env so QDRANT_HOST / QDRANT_PORT / QDRANT_COLLECTION are picked up
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants — match the existing API
# ---------------------------------------------------------------------------
COLUMNS = [
    "ID",
    "Name",
    "Latitude / Longitude",
    "Opening time",
    "Closing time",
    "Category",
    "Sub-category",
    "Estimated visit duration",
    "Indoor / outdoor",
    "Price range",
]

VECTOR_SIZE = 384          # all-MiniLM-L6-v2 output dimension
ST_MODEL_NAME = "all-MiniLM-L6-v2"

QDRANT_HOST       = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME   = os.environ.get("QDRANT_COLLECTION", "pois")

# ---------------------------------------------------------------------------
# Colour palette & fonts (modern dark theme)
# ---------------------------------------------------------------------------
BG_DARK     = "#0f1117"
BG_CARD     = "#1a1d27"
BG_INPUT    = "#252836"
ACCENT      = "#6c63ff"
ACCENT_LITE = "#9d97ff"
TEXT_MAIN   = "#e8e8f0"
TEXT_MUTED  = "#8a8aaa"
SUCCESS     = "#4caf82"
WARNING     = "#f0a500"
ERROR_COLOR = "#f05050"
BORDER      = "#2e3150"

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEADER = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 9)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_sentence(row: dict) -> str:
    """Convert a row dict into an embeddable natural-language sentence."""
    name        = row.get("Name", "Unknown")
    cat         = row.get("Category", "")
    sub         = row.get("Sub-category", "")
    indoor      = row.get("Indoor / outdoor", "")
    open_time   = row.get("Opening time", "")
    close_time  = row.get("Closing time", "")
    price_range = row.get("Price range", "")
    duration    = row.get("Estimated visit duration", "")
    location    = row.get("Latitude / Longitude", "")

    # Map price range symbols to readable text
    price_labels = {"$": "Budget", "$$": "Moderate", "$$$": "Luxury"}
    price_text = price_labels.get(str(price_range).strip(), str(price_range).strip())

    parts = [f"{name} is a {cat}"]
    if sub:
        parts[0] += f" ({sub})"
    parts[0] += "."
    if indoor:
        parts.append(f"It is {indoor}.")
    if duration:
        parts.append(f"Estimated visit duration: {duration}.")
    if open_time or close_time:
        parts.append(f"Opening hours: {open_time} - {close_time}.")
    if price_text:
        parts.append(f"Price range: {price_text}.")
    if location:
        parts.append(f"Location: {location}.")
    return " ".join(parts)


def safe_int_id(raw_id) -> int | None:
    """Return an integer Qdrant point ID or None (will then use UUID)."""
    try:
        return int(float(str(raw_id)))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class DataIngestionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("POI Data Ingestion — Qdrant")
        self.geometry("900x720")
        self.minsize(820, 640)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Shared state
        self._manual_rows: list[dict] = []   # rows from manual entry tab
        self._excel_df: pd.DataFrame | None = None  # dataframe from Excel tab
        self._st_model: SentenceTransformer | None = None
        self._qdrant: QdrantClient | None = None
        self._ready = False  # True once ST model + Qdrant connected

        self._build_ui()
        self._start_init()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_DARK, pady=18)
        header.pack(fill="x", padx=30)

        tk.Label(
            header, text="POI Data Ingestion", font=FONT_TITLE,
            bg=BG_DARK, fg=TEXT_MAIN,
        ).pack(side="left")

        self._status_dot = tk.Label(
            header, text="● Initialising…", font=FONT_BODY,
            bg=BG_DARK, fg=WARNING,
        )
        self._status_dot.pack(side="right", padx=6)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=0)

        # ── Notebook (tabs) ───────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",         background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",     background=BG_CARD, foreground=TEXT_MUTED,
                        font=FONT_HEADER, padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        style.configure("TFrame",            background=BG_DARK)
        style.configure("Treeview",          background=BG_INPUT, fieldbackground=BG_INPUT,
                        foreground=TEXT_MAIN, rowheight=26, font=FONT_BODY)
        style.configure("Treeview.Heading",  background=BG_CARD, foreground=ACCENT_LITE,
                        font=FONT_HEADER)
        style.map("Treeview",                background=[("selected", ACCENT)])
        style.configure("TProgressbar",      troughcolor=BG_CARD, background=ACCENT,
                        thickness=10)
        style.configure("TScrollbar",        background=BG_CARD, troughcolor=BG_DARK)

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=20, pady=(12, 0))

        # Tab 1 — Manual Entry
        tab1 = ttk.Frame(self._nb)
        self._nb.add(tab1, text="  ✏️  Manual Entry  ")
        self._build_manual_tab(tab1)

        # Tab 2 — Excel Upload
        tab2 = ttk.Frame(self._nb)
        self._nb.add(tab2, text="  📂  Excel Upload  ")
        self._build_excel_tab(tab2)

        # ── Bottom panel (shared) ─────────────────────────────────────
        bottom = tk.Frame(self, bg=BG_DARK, pady=10)
        bottom.pack(fill="x", padx=20, pady=(8, 0))

        btn_frame = tk.Frame(bottom, bg=BG_DARK)
        btn_frame.pack(fill="x")

        self._proceed_btn = tk.Button(
            btn_frame,
            text="  ▶  Proceed",
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="#ffffff",
            activebackground=ACCENT_LITE, activeforeground="#ffffff",
            relief="flat", bd=0, padx=28, pady=10, cursor="hand2",
            command=self._on_proceed,
        )
        self._proceed_btn.pack(side="right")

        self._clear_btn = tk.Button(
            btn_frame,
            text="  🗑  Clear Log",
            font=FONT_BODY,
            bg=BG_CARD, fg=TEXT_MUTED,
            activebackground=BORDER, activeforeground=TEXT_MAIN,
            relief="flat", bd=0, padx=14, pady=10, cursor="hand2",
            command=self._clear_log,
        )
        self._clear_btn.pack(side="right", padx=(0, 10))

        # Progress bar
        self._progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self._progress.pack(fill="x", padx=20, pady=(6, 0))

        # Log console
        log_frame = tk.Frame(self, bg=BG_DARK)
        log_frame.pack(fill="both", expand=False, padx=20, pady=(6, 14))

        self._log = tk.Text(
            log_frame, height=8, bg=BG_CARD, fg=TEXT_MAIN,
            font=FONT_MONO, relief="flat", state="disabled",
            wrap="word", bd=0, padx=10, pady=8,
            insertbackground=TEXT_MAIN,
        )
        log_scroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)

        # Tag colours for log
        self._log.tag_configure("info",    foreground=TEXT_MAIN)
        self._log.tag_configure("success", foreground=SUCCESS)
        self._log.tag_configure("warn",    foreground=WARNING)
        self._log.tag_configure("error",   foreground=ERROR_COLOR)
        self._log.tag_configure("accent",  foreground=ACCENT_LITE)

    # ── Manual Entry Tab ──────────────────────────────────────────────

    def _build_manual_tab(self, parent):
        parent.configure(style="TFrame")

        # Scrollable form area
        canvas = tk.Canvas(parent, bg=BG_DARK, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        form_frame = tk.Frame(canvas, bg=BG_DARK)

        form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        tk.Label(
            form_frame, text="Enter POI Details",
            font=FONT_HEADER, bg=BG_DARK, fg=ACCENT_LITE, pady=6,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(14, 4))

        self._entries: dict[str, tk.StringVar] = {}
        for i, col in enumerate(COLUMNS, start=1):
            tk.Label(
                form_frame, text=col, font=FONT_BODY,
                bg=BG_DARK, fg=TEXT_MUTED, anchor="e", width=26,
            ).grid(row=i, column=0, padx=(20, 10), pady=5, sticky="e")

            var = tk.StringVar()
            self._entries[col] = var
            entry = tk.Entry(
                form_frame, textvariable=var,
                font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN,
                insertbackground=TEXT_MAIN, relief="flat", bd=0,
                highlightthickness=1, highlightcolor=ACCENT,
                highlightbackground=BORDER,
            )
            entry.grid(row=i, column=1, padx=(0, 20), pady=5, sticky="ew", ipady=6)
            form_frame.columnconfigure(1, weight=1)

        # Add Row button
        add_btn = tk.Button(
            form_frame, text="  ＋  Add Row",
            font=FONT_BODY, bg=SUCCESS, fg="#ffffff",
            activebackground="#3da06a", activeforeground="#ffffff",
            relief="flat", bd=0, padx=18, pady=8, cursor="hand2",
            command=self._add_manual_row,
        )
        add_btn.grid(row=len(COLUMNS) + 1, column=1, sticky="e",
                     padx=20, pady=(10, 4))

        # Preview treeview
        tk.Label(
            form_frame, text="Added Rows Preview",
            font=FONT_HEADER, bg=BG_DARK, fg=ACCENT_LITE,
        ).grid(row=len(COLUMNS) + 2, column=0, columnspan=2,
               sticky="w", padx=20, pady=(12, 4))

        tree_frame = tk.Frame(form_frame, bg=BG_DARK)
        tree_frame.grid(row=len(COLUMNS) + 3, column=0, columnspan=2,
                        sticky="nsew", padx=20, pady=(0, 14))
        form_frame.rowconfigure(len(COLUMNS) + 3, weight=1)

        preview_cols = ["Name", "Category", "Sub-category", "Indoor / outdoor"]
        self._manual_tree = ttk.Treeview(
            tree_frame, columns=preview_cols, show="headings", height=5,
        )
        for col in preview_cols:
            self._manual_tree.heading(col, text=col)
            self._manual_tree.column(col, width=160, anchor="w")

        manual_yscroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                       command=self._manual_tree.yview)
        self._manual_tree.configure(yscrollcommand=manual_yscroll.set)
        manual_yscroll.pack(side="right", fill="y")
        self._manual_tree.pack(side="left", fill="both", expand=True)

        # Counter label
        self._row_count_label = tk.Label(
            form_frame, text="0 rows queued",
            font=FONT_BODY, bg=BG_DARK, fg=TEXT_MUTED,
        )
        self._row_count_label.grid(row=len(COLUMNS) + 4, column=0,
                                   columnspan=2, sticky="w", padx=20, pady=(0, 8))

    # ── Excel Upload Tab ──────────────────────────────────────────────

    def _build_excel_tab(self, parent):
        parent.configure(style="TFrame")

        top = tk.Frame(parent, bg=BG_DARK, pady=16)
        top.pack(fill="x", padx=20)

        tk.Label(
            top, text="Upload Excel File",
            font=FONT_HEADER, bg=BG_DARK, fg=ACCENT_LITE,
        ).pack(side="left")

        self._browse_btn = tk.Button(
            top, text="  📂  Browse…",
            font=FONT_BODY, bg=BG_CARD, fg=TEXT_MAIN,
            activebackground=BORDER, activeforeground=TEXT_MAIN,
            relief="flat", bd=0, padx=14, pady=8, cursor="hand2",
            command=self._browse_excel,
        )
        self._browse_btn.pack(side="right")

        # File path label
        self._file_label = tk.Label(
            parent, text="No file selected",
            font=FONT_BODY, bg=BG_DARK, fg=TEXT_MUTED, anchor="w",
        )
        self._file_label.pack(fill="x", padx=20, pady=(0, 8))

        # Info frame
        info_frame = tk.Frame(parent, bg=BG_CARD, bd=0,
                              highlightthickness=1, highlightbackground=BORDER)
        info_frame.pack(fill="x", padx=20, pady=(0, 12))

        self._excel_info = tk.Label(
            info_frame,
            text="Select an Excel file to see a preview.",
            font=FONT_BODY, bg=BG_CARD, fg=TEXT_MUTED,
            anchor="w", padx=14, pady=10, justify="left",
        )
        self._excel_info.pack(fill="x")

        # Preview treeview (first 10 rows)
        tk.Label(
            parent, text="Preview (first 10 rows)",
            font=FONT_HEADER, bg=BG_DARK, fg=ACCENT_LITE,
        ).pack(anchor="w", padx=20, pady=(4, 4))

        tree_wrap = tk.Frame(parent, bg=BG_DARK)
        tree_wrap.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        self._excel_tree = ttk.Treeview(tree_wrap, columns=COLUMNS, show="headings", height=10)
        for col in COLUMNS:
            self._excel_tree.heading(col, text=col)
            w = 60 if col == "ID" else 140
            self._excel_tree.column(col, width=w, anchor="w", minwidth=50)

        ex_xscroll = ttk.Scrollbar(tree_wrap, orient="horizontal",
                                   command=self._excel_tree.xview)
        ex_yscroll = ttk.Scrollbar(tree_wrap, orient="vertical",
                                   command=self._excel_tree.yview)
        self._excel_tree.configure(xscrollcommand=ex_xscroll.set,
                                   yscrollcommand=ex_yscroll.set)
        ex_yscroll.pack(side="right", fill="y")
        ex_xscroll.pack(side="bottom", fill="x")
        self._excel_tree.pack(side="left", fill="both", expand=True)

    # ------------------------------------------------------------------
    # Initialisation (background thread)
    # ------------------------------------------------------------------

    def _start_init(self):
        self._log_msg("⚙️  Loading Sentence Transformer model…", "accent")
        threading.Thread(target=self._init_models, daemon=True).start()

    def _init_models(self):
        try:
            self._log_msg(f"   Model: {ST_MODEL_NAME}", "info")
            self._st_model = SentenceTransformer(ST_MODEL_NAME)

            self._log_msg(
                f"   Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}…", "info"
            )
            self._qdrant = QdrantClient(
                host=QDRANT_HOST, port=QDRANT_PORT,
                https=(QDRANT_PORT == 443),
            )
            # Ping
            self._qdrant.get_collections()

            self._ready = True
            self.after(0, self._on_init_done)
        except Exception as exc:
            self.after(0, lambda: self._on_init_error(str(exc)))

    def _on_init_done(self):
        self._status_dot.configure(text="● Ready", fg=SUCCESS)
        self._log_msg(
            f"✅  Ready — Qdrant collection: '{COLLECTION_NAME}'", "success"
        )

    def _on_init_error(self, msg: str):
        self._status_dot.configure(text="● Error", fg=ERROR_COLOR)
        self._log_msg(f"❌  Initialisation failed: {msg}", "error")
        messagebox.showerror(
            "Initialisation Error",
            f"Could not connect to Qdrant or load the model.\n\n{msg}\n\n"
            "Make sure Docker is running:\n"
            "  docker run -d -p 6333:6333 qdrant/qdrant",
        )

    # ------------------------------------------------------------------
    # Manual entry actions
    # ------------------------------------------------------------------

    def _add_manual_row(self):
        row = {col: self._entries[col].get().strip() for col in COLUMNS}
        if not row["Name"]:
            messagebox.showwarning("Missing Field", "The 'Name' field is required.")
            return

        self._manual_rows.append(row)
        # Insert into preview tree
        self._manual_tree.insert(
            "", "end",
            values=(
                row.get("Name", ""),
                row.get("Category", ""),
                row.get("Sub-category", ""),
                row.get("Indoor / outdoor", ""),
            ),
        )
        self._row_count_label.configure(
            text=f"{len(self._manual_rows)} row(s) queued"
        )
        # Clear form
        for var in self._entries.values():
            var.set("")
        self._log_msg(f"   ➕ Row added: {row['Name']}", "info")

    # ------------------------------------------------------------------
    # Excel browsing
    # ------------------------------------------------------------------

    def _browse_excel(self):
        path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_excel(path)
            # Normalise columns: strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            # Align to expected columns (add missing ones as empty)
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[COLUMNS]
            self._excel_df = df

            self._file_label.configure(text=f"📄  {path}", fg=TEXT_MAIN)
            found = [c for c in COLUMNS if c in df.columns]
            missing = [c for c in COLUMNS if c not in found]

            info_text = (
                f"Rows: {len(df)}    Columns matched: {len(found)}/10"
                + (f"    ⚠ Missing: {', '.join(missing)}" if missing else "")
            )
            self._excel_info.configure(text=info_text, fg=TEXT_MAIN)

            # Populate preview tree
            for row in self._excel_tree.get_children():
                self._excel_tree.delete(row)
            for _, row in df.head(10).iterrows():
                self._excel_tree.insert(
                    "", "end",
                    values=tuple(str(row.get(col, "")) for col in COLUMNS),
                )
            self._log_msg(f"📂  Loaded {len(df)} rows from {path}", "accent")
        except Exception as exc:
            messagebox.showerror("File Error", f"Could not read the file:\n{exc}")
            self._log_msg(f"❌  Failed to load Excel: {exc}", "error")

    # ------------------------------------------------------------------
    # Proceed
    # ------------------------------------------------------------------

    def _on_proceed(self):
        if not self._ready:
            messagebox.showwarning(
                "Not Ready",
                "The model and Qdrant are still initialising. Please wait.",
            )
            return

        active_tab = self._nb.index(self._nb.select())

        if active_tab == 0:  # Manual Entry
            if not self._manual_rows:
                messagebox.showwarning(
                    "No Data",
                    "Please add at least one row using the '+ Add Row' button.",
                )
                return
            rows = list(self._manual_rows)
        else:  # Excel Upload
            if self._excel_df is None:
                messagebox.showwarning(
                    "No File",
                    "Please select an Excel file first.",
                )
                return
            rows = self._excel_df.to_dict(orient="records")

        # Disable button during ingestion
        self._proceed_btn.configure(state="disabled", text="  ⏳  Processing…")
        self._progress["value"] = 0
        self._progress["maximum"] = len(rows)

        threading.Thread(
            target=self._ingest_rows, args=(rows,), daemon=True
        ).start()

    def _ingest_rows(self, rows: list[dict]):
        """Embed and upsert rows — runs in a background thread."""
        self._log_msg(
            f"\n🚀  Starting ingestion of {len(rows)} row(s)…", "accent"
        )

        # Ensure collection exists
        try:
            existing = [c.name for c in self._qdrant.get_collections().collections]
            if COLLECTION_NAME not in existing:
                self._log_msg(
                    f"   Creating collection '{COLLECTION_NAME}'…", "info"
                )
                self._qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
                self._log_msg(f"   ✅ Collection created.", "success")
            else:
                self._log_msg(
                    f"   Collection '{COLLECTION_NAME}' already exists.", "info"
                )
        except Exception as exc:
            self._log_msg(f"❌  Collection setup failed: {exc}", "error")
            self.after(0, self._reset_proceed_btn)
            return

        success_count = 0
        error_count = 0

        for i, row in enumerate(rows, start=1):
            try:
                sentence = build_sentence(row)
                vector = self._st_model.encode(sentence).tolist()

                raw_id = row.get("ID", "")
                point_id = safe_int_id(raw_id)
                if point_id is None:
                    point_id = str(uuid.uuid4())

                # Clean payload — convert everything to str to stay JSON-safe
                payload = {k: str(v) if v is not None else "" for k, v in row.items()}

                self._qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
                success_count += 1
                if i % 10 == 0 or i == len(rows):
                    self._log_msg(
                        f"   [{i}/{len(rows)}] ✔ Upserted: {row.get('Name', '?')}", "info"
                    )
            except Exception as exc:
                error_count += 1
                self._log_msg(
                    f"   [{i}/{len(rows)}] ✘ Error for '{row.get('Name', '?')}': {exc}",
                    "error",
                )

            # Update progress bar on main thread
            self.after(0, lambda v=i: self._progress.configure(value=v))

        summary = (
            f"\n✅  Ingestion complete — {success_count} succeeded"
            + (f", {error_count} failed" if error_count else "")
            + f".\n   Qdrant dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard"
        )
        self._log_msg(summary, "success" if error_count == 0 else "warn")
        self.after(0, self._reset_proceed_btn)

        # Clear manual rows after successful ingestion
        if success_count > 0 and self._nb.index(self._nb.select()) == 0:
            self._manual_rows.clear()
            self.after(0, self._clear_manual_preview)

    def _clear_manual_preview(self):
        for item in self._manual_tree.get_children():
            self._manual_tree.delete(item)
        self._row_count_label.configure(text="0 rows queued")

    def _reset_proceed_btn(self):
        self._proceed_btn.configure(state="normal", text="  ▶  Proceed")

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str, tag: str = "info"):
        def _append():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n", tag)
            self._log.see("end")
            self._log.configure(state="disabled")
        # Safe to call from any thread
        self.after(0, _append)

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._progress["value"] = 0


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = DataIngestionApp()
    app.mainloop()
