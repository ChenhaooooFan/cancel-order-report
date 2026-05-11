import html
import re
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="Cancelled + Returned + Auction Orders Report Generator",
    page_icon="💅",
    layout="wide",
)


# ============================================================
# Constants
# ============================================================
CANCELLED_STATUSES = {"cancelled", "canceled"}
SIZE_TOKENS = {
    "XS", "S", "M", "L", "XL", "XXL", "XXXL",
    "EXTRA SMALL", "SMALL", "MEDIUM", "LARGE", "EXTRA LARGE",
}
RETURN_TARGET_PRODUCT_LINKS = [
    "Dreamwear",
    "Top Trend",
    "Next Gen",
    "New Drop",
    "Final Sale",
    "Square",
    "Stiletto",
    "Almond",
    "Spring Bloom",
    "Summer Shine",
    "Best Seller",
    "Organizer Binder",
    "TOOLKITS",
    "BUY 4 GET 1 FREE",
]


# ============================================================
# Basic helpers
# ============================================================
def clean_column_name(col) -> str:
    return str(col).replace("\ufeff", "").strip()


def normalize_text(x, default="Unknown") -> str:
    if pd.isna(x):
        return default
    s = str(x).replace("\t", "").replace("\r", " ").replace("\n", " ").strip()
    if not s or s.lower() in {"nan", "nat", "none"}:
        return default
    return s


def stringify_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\t", "").replace("\r", "").replace("\n", "").strip()
    if s.endswith(".0") and s[:-2].replace(".", "", 1).isdigit():
        s = s[:-2]
    return s


def parse_datetime_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace("\t", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "None": np.nan})
    )
    # TikTok order exports are usually MM/DD/YYYY HH:MM:SS AM/PM.
    parsed = pd.to_datetime(cleaned, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    missing = parsed.isna() & cleaned.notna()
    if missing.any():
        # Return/Refund exports sometimes use DD/MM/YYYY HH:MM:SS.
        parsed.loc[missing] = pd.to_datetime(cleaned.loc[missing], dayfirst=True, errors="coerce")
    missing = parsed.isna() & cleaned.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(cleaned.loc[missing], errors="coerce")
    return parsed


def parse_number_series(s: pd.Series, default=np.nan) -> pd.Series:
    out = pd.to_numeric(
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.strip(),
        errors="coerce",
    )
    if not pd.isna(default):
        out = out.fillna(default)
    return out


def pct(n, d, digits=1) -> float:
    if d == 0 or pd.isna(d):
        return 0.0
    return round(float(n) / float(d) * 100, digits)


def fmt_num(x, digits=0) -> str:
    if pd.isna(x):
        return "-"
    if digits == 0:
        return f"{int(round(float(x))):,}"
    return f"{float(x):,.{digits}f}"


def fmt_money(x, digits=2) -> str:
    if pd.isna(x):
        return "-"
    return f"${float(x):,.{digits}f}"


def fmt_pct(n, d, digits=1) -> str:
    return f"{pct(n, d, digits):.{digits}f}%"


def first_non_null(x):
    x = x.dropna()
    return x.iloc[0] if len(x) else pd.NaT


def mode_or_first(x):
    x = x.dropna()
    if len(x) == 0:
        return "Unknown"
    mode = x.mode()
    return mode.iloc[0] if len(mode) else x.iloc[0]


def join_unique(x, max_items=30):
    vals = []
    for v in x:
        s = normalize_text(v, default="")
        if s and s not in vals:
            vals.append(s)
    if len(vals) > max_items:
        vals = vals[:max_items] + [f"...+{len(vals)-max_items} more"]
    return "; ".join(vals) if vals else "Unknown"


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        data = uploaded_file.getvalue()
        for enc in ["utf-8-sig", "utf-8", "gb18030", "latin1"]:
            try:
                return pd.read_csv(BytesIO(data), dtype=str, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(BytesIO(data), dtype=str)
    return pd.read_excel(uploaded_file, dtype=str)


def clean_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]
    return out


def contains_return_refund(x) -> bool:
    s = normalize_text(x, default="").lower()
    return "return" in s or "refund" in s


def strip_seller_sku_size_suffix(x) -> str:
    s = normalize_text(x, default="Unknown")
    if s == "Unknown":
        return s
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"(?i)([-_\s]+)(XS|S|M|L|XL|XXL|XXXL)$", "", s)
    return s or "Unknown"


def strip_sku_name_size_suffix(x) -> str:
    s = normalize_text(x, default="Unknown")
    if s == "Unknown":
        return s
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2 and parts[-1].upper() in SIZE_TOKENS:
        s = ", ".join(parts[:-1]).strip()
    s = re.sub(r"(?i)\s*[-_/]\s*(XS|S|M|L|XL|XXL|XXXL)$", "", s).strip()
    return s or "Unknown"


def variation_to_style(row) -> str:
    variation = normalize_text(row.get("Variation", ""), default="")
    if variation:
        parts = [p.strip() for p in variation.split(",") if p.strip()]
        if len(parts) >= 2 and parts[-1].upper() in SIZE_TOKENS:
            return ", ".join(parts[:-1]).strip() or variation
        return variation
    return normalize_text(row.get("Product Name", ""), default="Unknown")


def classify_live_segment(hour, live1_start, live1_end, live2_start, live2_end) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if live1_start <= h < live1_end:
        return "直播①"
    if live2_start <= h < live2_end:
        return "直播②"
    return "非直播"


def hourly_counts(series: pd.Series) -> list[int]:
    if series is None or len(series) == 0:
        return [0] * 24
    s = series.dropna().dt.hour.value_counts().sort_index()
    return [int(s.get(h, 0)) for h in range(24)]


def peak_label(counts: list[int]) -> tuple[str, int]:
    if not counts or max(counts) == 0:
        return "-", 0
    maxv = max(counts)
    hrs = [i for i, v in enumerate(counts) if v == maxv]
    if len(hrs) == 1:
        return f"{hrs[0]}点", maxv
    if len(hrs) <= 3:
        return " & ".join([f"{h}点" for h in hrs]), maxv
    return f"{hrs[0]}点等", maxv


# ============================================================
# Catalog mapping
# ============================================================
def build_catalog_sku_name_map(catalog_raw: pd.DataFrame | None) -> dict:
    if catalog_raw is None or catalog_raw.empty:
        return {}
    df = clean_df_columns(catalog_raw)
    if "SKU" not in df.columns or "款式英文名称" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        sku = strip_seller_sku_size_suffix(row.get("SKU", ""))
        name = normalize_text(row.get("款式英文名称", ""), default="")
        if sku and sku != "Unknown" and name:
            out[sku] = name
    return out


# ============================================================
# All-order / Cancelled order preparation
# ============================================================
def validate_all_order_columns(raw: pd.DataFrame) -> list[str]:
    required = ["Order ID", "Order Status", "Created Time", "Product Name"]
    return [c for c in required if c not in raw.columns]


def prepare_order_lines(raw: pd.DataFrame, metric_mode="quantity") -> pd.DataFrame:
    df = clean_df_columns(raw)
    if "Order ID" in df.columns:
        df["Order ID"] = df["Order ID"].apply(stringify_id)
    if "Order Status" in df.columns:
        df["Order Status Clean"] = df["Order Status"].apply(lambda x: normalize_text(x, default="Unknown"))
    else:
        df["Order Status Clean"] = "Unknown"

    df["Is Cancelled Line"] = df["Order Status Clean"].str.lower().isin(CANCELLED_STATUSES)
    df["Created Datetime"] = parse_datetime_series(df["Created Time"]) if "Created Time" in df.columns else pd.NaT
    df["Cancelled Datetime"] = parse_datetime_series(df["Cancelled Time"]) if "Cancelled Time" in df.columns else pd.NaT
    df["Cancel Reason Clean"] = df["Cancel Reason"].apply(lambda x: normalize_text(x, default="Unknown")) if "Cancel Reason" in df.columns else "Unknown"
    df["Product Link / Product Name"] = df["Product Name"].apply(lambda x: normalize_text(x, default="Unknown")) if "Product Name" in df.columns else "Unknown"
    df["Nail Style"] = df.apply(variation_to_style, axis=1)
    df["Seller SKU Base"] = df["Seller SKU"].apply(strip_seller_sku_size_suffix) if "Seller SKU" in df.columns else "Unknown"
    df["SKU Row Count"] = 1
    df["Quantity Parsed"] = parse_number_series(df["Quantity"], default=1) if "Quantity" in df.columns else 1
    df["Return Quantity in All Order Parsed"] = parse_number_series(df["Sku Quantity of return"], default=0) if "Sku Quantity of return" in df.columns else 0
    df["Metric Units"] = df["Quantity Parsed"] if metric_mode == "quantity" else df["SKU Row Count"]
    if "Order Amount" in df.columns:
        df["Order Amount Parsed"] = parse_number_series(df["Order Amount"], default=0)
    if "Order Refund Amount" in df.columns:
        df["Order Refund Amount Parsed"] = parse_number_series(df["Order Refund Amount"], default=0)
    return df


def filter_by_created_date(lines: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    df = lines[lines["Created Datetime"].notna()].copy()
    df["Created Date"] = df["Created Datetime"].dt.date
    return df[(df["Created Date"] >= start_date) & (df["Created Date"] <= end_date)].copy()


def build_order_level(lines: pd.DataFrame, live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    tmp = lines.copy()
    tmp["__cancel_int"] = tmp["Is Cancelled Line"].astype(int)
    tmp["__return_refund_int"] = tmp.get("Cancelation/Return Type", pd.Series("", index=tmp.index)).apply(contains_return_refund).astype(int)
    agg = {
        "Created Datetime": "min",
        "Cancelled Datetime": first_non_null,
        "Order Status Clean": mode_or_first,
        "__cancel_int": "max",
        "__return_refund_int": "max",
        "Cancel Reason Clean": mode_or_first,
        "Nail Style": join_unique,
        "Product Link / Product Name": join_unique,
        "Metric Units": "sum",
        "SKU Row Count": "sum",
        "Quantity Parsed": "sum",
        "Return Quantity in All Order Parsed": "sum",
        "Seller SKU Base": join_unique,
    }
    for c in ["Seller SKU", "Variation", "Cancelation/Return Type", "Order Amount Parsed", "Order Refund Amount Parsed"]:
        if c in tmp.columns:
            agg[c] = first_non_null if c.endswith("Parsed") else join_unique
    od = tmp.groupby("Order ID", as_index=False).agg(agg)
    od["Is Cancelled"] = od["__cancel_int"].eq(1)
    od["Has Return Refund in All Order"] = od["__return_refund_int"].eq(1)
    od["Created Hour"] = od["Created Datetime"].dt.hour
    od["Created Day Type"] = np.where(od["Created Datetime"].dt.weekday >= 5, "周末 Weekend", "工作日 Weekday")
    od["Live Segment by Created Time"] = od["Created Hour"].apply(lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end))
    od["Is Live by Created Time"] = od["Live Segment by Created Time"].isin(["直播①", "直播②"])
    od["Cancelled Hour"] = od["Cancelled Datetime"].dt.hour
    od["Cancelled Day Type"] = np.where(od["Cancelled Datetime"].dt.weekday >= 5, "周末 Weekend", "工作日 Weekday")
    return od.drop(columns=["__cancel_int", "__return_refund_int"])


def make_breakdown(lines: pd.DataFrame, group_col: str, top_n: int, metric_col="Metric Units") -> pd.DataFrame:
    if lines.empty or group_col not in lines.columns:
        return pd.DataFrame(columns=[group_col, "Units", "Order Count", "Quantity", "占比 %"])
    total = lines[metric_col].sum()
    out = (
        lines.groupby(group_col, dropna=False)
        .agg(
            Units=(metric_col, "sum"),
            **{"Order Count": ("Order ID", "nunique")},
            Quantity=("Quantity Parsed", "sum"),
        )
        .reset_index()
    )
    out[group_col] = out[group_col].apply(lambda x: normalize_text(x, default="Unknown"))
    out["占比 %"] = out["Units"].apply(lambda x: pct(x, total))
    return out.sort_values(["Units", "Order Count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def value_count_table(df: pd.DataFrame, col: str, denominator: int, top_n=10, count_label="Order Count") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, count_label, "占比 %"])
    out = df[col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(top_n).reset_index()
    out.columns = [col, count_label]
    out["占比 %"] = out[count_label].apply(lambda x: pct(x, denominator))
    return out


# ============================================================
# Returned order preparation
# ============================================================
def is_no_longer_needed(reason) -> bool:
    return normalize_text(reason, default="").strip().lower() == "no longer needed"


def is_request_cancelled_status(x) -> bool:
    s = normalize_text(x, default="").lower()
    return "request" in s and ("cancelled" in s or "canceled" in s or "cancel" in s)


def is_refund_only_type(x) -> bool:
    s = normalize_text(x, default="").lower().replace("-", " ").strip()
    return s == "refund only"


def choose_return_package_key(row) -> str:
    tracking = normalize_text(row.get("Return Logistics Tracking ID", ""), default="")
    return_order_id = normalize_text(row.get("Return Order ID", ""), default="")
    order_id = normalize_text(row.get("Order ID", ""), default="")
    if tracking:
        return f"TRACKING：{tracking}"
    if return_order_id:
        return f"RETURN_ORDER：{return_order_id}"
    if order_id:
        return f"ORDER：{order_id}"
    return "UNKNOWN_PACKAGE"


def prepare_return_lines(return_raw: pd.DataFrame, all_order_lookup: pd.DataFrame, catalog_map: dict,
                         live1_start, live1_end, live2_start, live2_end) -> pd.DataFrame:
    df = clean_df_columns(return_raw)
    if df.empty:
        return df
    if "Order ID" in df.columns:
        df["Order ID"] = df["Order ID"].apply(stringify_id)
    if "Return Order ID" in df.columns:
        df["Return Order ID"] = df["Return Order ID"].apply(stringify_id)

    for c in ["Return Logistics Tracking ID", "Return Reason", "Return Type", "Return Status", "Return Sub Status", "Product Name", "SKU Name", "Seller SKU"]:
        if c not in df.columns:
            df[c] = ""

    df["Return Quantity Parsed"] = parse_number_series(df["Return Quantity"], default=1) if "Return Quantity" in df.columns else 1
    df["Returned SKU Row Count"] = 1
    df["Return Package Key"] = df.apply(choose_return_package_key, axis=1)
    df["Return Reason Clean"] = df["Return Reason"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Type Clean"] = df["Return Type"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Sub Status Clean"] = df["Return Sub Status"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return Product Link (I Product Name)"] = df["Product Name"].apply(lambda x: normalize_text(x, default="Unknown"))
    df["Return SKU (H no size)"] = df["Seller SKU"].apply(strip_seller_sku_size_suffix)
    df["Return Style (J SKU Name)"] = df["SKU Name"].apply(strip_sku_name_size_suffix)
    df["Return Style English (Catalog)"] = df["Return SKU (H no size)"].map(catalog_map).fillna(df["Return Style (J SKU Name)"])
    df["Return Style English (Catalog)"] = df["Return Style English (Catalog)"].apply(lambda x: normalize_text(x, default="Unknown"))

    df["Seller Fault Flag"] = ~df["Return Reason Clean"].apply(is_no_longer_needed)
    df["Request Cancelled Flag"] = df["Return Sub Status Clean"].apply(is_request_cancelled_status)
    df["Has Return Tracking Flag"] = df["Return Logistics Tracking ID"].apply(lambda x: bool(normalize_text(x, default="")))
    df["Refund Only Flag"] = df["Return Type Clean"].apply(is_refund_only_type)

    lookup = all_order_lookup[["Order ID", "Created Datetime"]].drop_duplicates("Order ID") if not all_order_lookup.empty else pd.DataFrame(columns=["Order ID", "Created Datetime"])
    df = df.merge(lookup, on="Order ID", how="left")
    if "Created Time" in df.columns:
        df["Created Datetime"] = df["Created Datetime"].fillna(parse_datetime_series(df["Created Time"]))
    df["Created Hour"] = df["Created Datetime"].dt.hour
    df["Live Segment by Created Time"] = df["Created Hour"].apply(lambda h: classify_live_segment(h, live1_start, live1_end, live2_start, live2_end))
    df.loc[df["Created Datetime"].isna(), "Live Segment by Created Time"] = "Unknown"
    df["Is Live by Created Time"] = df["Live Segment by Created Time"].isin(["直播①", "直播②"])
    return df


def build_return_package_level(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame()
    tmp = lines.copy()
    for flag in ["Seller Fault Flag", "Request Cancelled Flag", "Has Return Tracking Flag", "Refund Only Flag"]:
        tmp[flag + " Int"] = tmp[flag].astype(int)
    pkg = (
        tmp.groupby("Return Package Key", as_index=False)
        .agg(
            **{
                "Order IDs": ("Order ID", join_unique),
                "Return Order IDs": ("Return Order ID", join_unique),
                "Created Datetime": ("Created Datetime", first_non_null),
                "Return Reason": ("Return Reason Clean", mode_or_first),
                "Return Type": ("Return Type Clean", mode_or_first),
                "Return Sub Status": ("Return Sub Status Clean", mode_or_first),
                "Return Logistics Tracking ID": ("Return Logistics Tracking ID", mode_or_first),
                "Returned Units": ("Return Quantity Parsed", "sum"),
                "SKU Rows": ("Returned SKU Row Count", "sum"),
                "Return Product Links": ("Return Product Link (I Product Name)", join_unique),
                "Return Styles": ("Return Style English (Catalog)", join_unique),
                "Seller Fault Int": ("Seller Fault Flag Int", "max"),
                "Request Cancelled Int": ("Request Cancelled Flag Int", "max"),
                "Has Return Tracking Int": ("Has Return Tracking Flag Int", "max"),
                "Refund Only Int": ("Refund Only Flag Int", "max"),
                "Live Segment by Created Time": ("Live Segment by Created Time", mode_or_first),
                "Is Live Int": ("Is Live by Created Time", "max"),
            }
        )
    )
    pkg["Seller Fault Flag"] = pkg["Seller Fault Int"].eq(1)
    pkg["Request Cancelled Flag"] = pkg["Request Cancelled Int"].eq(1)
    pkg["Has Return Tracking Flag"] = pkg["Has Return Tracking Int"].eq(1)
    pkg["Refund Only Flag"] = pkg["Refund Only Int"].eq(1)
    pkg["Return Reason Attribution"] = np.where(pkg["Seller Fault Flag"], "Seller Fault", "No Longer Needed")
    pkg["Created Hour"] = pkg["Created Datetime"].dt.hour
    pkg.loc[pkg["Created Datetime"].isna(), "Live Segment by Created Time"] = "Unknown"
    pkg["Is Live by Created Time"] = pkg["Is Live Int"].eq(1)
    return pkg.drop(columns=["Seller Fault Int", "Request Cancelled Int", "Has Return Tracking Int", "Refund Only Int", "Is Live Int"])


def make_return_line_breakdown(lines: pd.DataFrame, group_col: str, top_n=10) -> pd.DataFrame:
    if lines.empty or group_col not in lines.columns:
        return pd.DataFrame(columns=[group_col, "Returned Units", "Return Package Count", "SKU Rows", "占比 %"])
    total = lines["Return Quantity Parsed"].sum()
    out = (
        lines.groupby(group_col, dropna=False)
        .agg(
            **{
                "Returned Units": ("Return Quantity Parsed", "sum"),
                "Return Package Count": ("Return Package Key", "nunique"),
                "SKU Rows": ("Returned SKU Row Count", "sum"),
            }
        )
        .reset_index()
    )
    out[group_col] = out[group_col].apply(lambda x: normalize_text(x, default="Unknown"))
    out["占比 %"] = out["Returned Units"].apply(lambda x: pct(x, total))
    return out.sort_values(["Returned Units", "Return Package Count"], ascending=[False, False]).head(top_n).reset_index(drop=True)


def make_package_count_table(pkg: pd.DataFrame, col: str, top_n=10) -> pd.DataFrame:
    return value_count_table(pkg, col, len(pkg), top_n, count_label="Returned Packages")


def make_return_boolean_summary(pkg: pd.DataFrame) -> pd.DataFrame:
    total = len(pkg)
    rows = [
        ["Seller Fault", int(pkg["Seller Fault Flag"].sum()) if total else 0, "Return Reason 不是 No Longer Needed"],
        ["Request Cancelled", int(pkg["Request Cancelled Flag"].sum()) if total else 0, "S column Return Sub Status"],
        ["已寄出退回包裹", int(pkg["Has Return Tracking Flag"].sum()) if total else 0, "Q column Return Logistics Tracking ID 有记录"],
        ["Refund Only", int(pkg["Refund Only Flag"].sum()) if total else 0, "L column Return Type = Refund Only"],
    ]
    out = pd.DataFrame(rows, columns=["Metric", "Returned Packages", "Definition"])
    out["占比 %"] = out["Returned Packages"].apply(lambda x: pct(x, total))
    return out


def make_return_target_link_share(lines: pd.DataFrame) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["Product Link", "Returned Units", "Return Package Count", "SKU Rows", "占比 %"])
    total = lines["Return Quantity Parsed"].sum()
    product_lower = lines["Return Product Link (I Product Name)"].fillna("").astype(str).str.lower()
    rows = []
    for label in RETURN_TARGET_PRODUCT_LINKS:
        mask = product_lower.str.contains(re.escape(label.lower()), na=False, regex=True)
        sub = lines[mask]
        units = float(sub["Return Quantity Parsed"].sum()) if not sub.empty else 0
        pkg_count = int(sub["Return Package Key"].nunique()) if not sub.empty else 0
        sku_rows = int(sub["Returned SKU Row Count"].sum()) if not sub.empty else 0
        rows.append([label, units, pkg_count, sku_rows, pct(units, total)])
    return pd.DataFrame(rows, columns=["Product Link", "Returned Units", "Return Package Count", "SKU Rows", "占比 %"])


def build_return_metrics(return_lines: pd.DataFrame, top_n=10) -> dict:
    pkg = build_return_package_level(return_lines)
    total = len(pkg)
    live = int(pkg["Is Live by Created Time"].sum()) if total else 0
    unknown = int((pkg["Live Segment by Created Time"] == "Unknown").sum()) if total else 0
    segment = pd.DataFrame([
        ["直播①", int((pkg["Live Segment by Created Time"] == "直播①").sum()) if total else 0],
        ["直播②", int((pkg["Live Segment by Created Time"] == "直播②").sum()) if total else 0],
        ["直播合计", live],
        ["非直播", int((pkg["Live Segment by Created Time"] == "非直播").sum()) if total else 0],
        ["Unknown", unknown],
    ], columns=["Segment", "Returned Packages"])
    segment["占比 %"] = segment["Returned Packages"].apply(lambda x: pct(x, total))
    return {
        "lines": return_lines,
        "packages": pkg,
        "total_packages": total,
        "live_packages": live,
        "unknown_packages": unknown,
        "segment_summary": segment,
        "boolean_summary": make_return_boolean_summary(pkg) if total else pd.DataFrame(),
        "reason_df": make_package_count_table(pkg, "Return Reason", top_n),
        "fault_df": make_package_count_table(pkg, "Return Reason Attribution", top_n),
        "sku_top10": make_return_line_breakdown(return_lines, "Return Style English (Catalog)", 10),
        "style_j": make_return_line_breakdown(return_lines, "Return Style (J SKU Name)", top_n),
        "product_top5": make_return_line_breakdown(return_lines, "Return Product Link (I Product Name)", 5),
        "product_targets": make_return_target_link_share(return_lines),
        "return_type_df": make_package_count_table(pkg, "Return Type", top_n),
        "return_status_df": make_package_count_table(pkg, "Return Sub Status", top_n),
    }


# ============================================================
# Auction analysis
# ============================================================
def make_auction_return_lines_from_all_order(auction_lines: pd.DataFrame) -> pd.DataFrame:
    """Fallback return detail when no Returned Order table is uploaded.
    Uses Auction/All-order rows where Cancelation/Return Type contains Return/Refund.
    """
    if auction_lines.empty or "Cancelation/Return Type" not in auction_lines.columns:
        return pd.DataFrame()
    df = auction_lines[auction_lines["Cancelation/Return Type"].apply(contains_return_refund)].copy()
    if df.empty:
        return df
    df["Return Quantity Parsed"] = df["Return Quantity in All Order Parsed"].replace(0, np.nan).fillna(df["Quantity Parsed"]).fillna(1)
    df["Returned SKU Row Count"] = 1
    df["Return Package Key"] = df["Order ID"].apply(lambda x: f"ORDER：{x}")
    df["Return Reason Clean"] = "Return/Refund"
    df["Return Type Clean"] = "Return/Refund"
    df["Return Sub Status Clean"] = "Unknown"
    df["Return Product Link (I Product Name)"] = df["Product Link / Product Name"]
    df["Return SKU (H no size)"] = df["Seller SKU Base"]
    df["Return Style (J SKU Name)"] = df["Nail Style"]
    df["Return Style English (Catalog)"] = df["Nail Style"]
    df["Return Logistics Tracking ID"] = ""
    df["Return Order ID"] = ""
    df["Seller Fault Flag"] = True
    df["Request Cancelled Flag"] = False
    df["Has Return Tracking Flag"] = False
    df["Refund Only Flag"] = False
    df["Created Hour"] = df["Created Datetime"].dt.hour
    df["Live Segment by Created Time"] = df.get("Live Segment by Created Time", "Unknown")
    df["Is Live by Created Time"] = df.get("Is Live by Created Time", False)
    return df




def mean_order_amount(order_df: pd.DataFrame) -> float:
    """Average paid order amount at order level. Uses Z column / Order Amount after Order ID dedupe."""
    if order_df is None or order_df.empty or "Order Amount Parsed" not in order_df.columns:
        return np.nan
    amounts = pd.to_numeric(order_df["Order Amount Parsed"], errors="coerce").dropna()
    if amounts.empty:
        return np.nan
    return float(amounts.mean())


def make_auction_aov_summary(ctx: dict) -> pd.DataFrame:
    return pd.DataFrame([
        ["有效 Auction 订单数（已排除 Cancelled）", ctx.get("non_cancelled_orders", 0)],
        ["全部 Auction 平均 AOV（排除 Cancelled）", fmt_money(ctx.get("auction_aov_excl_cancelled", np.nan))],
        ["提交退货 Auction 订单数", ctx.get("return_orders", 0)],
        ["提交退货 Auction 平均 AOV", fmt_money(ctx.get("return_aov", np.nan))],
        ["退货 AOV - 全部有效 Auction AOV", fmt_money(ctx.get("aov_gap", np.nan))],
    ], columns=["指标", "数值"])

def build_auction_metrics(auction_raw: pd.DataFrame, all_order_lines: pd.DataFrame, return_lines: pd.DataFrame,
                          metric_mode: str, live1_start, live1_end, live2_start, live2_end,
                          start_date: date, end_date: date, top_n=10) -> dict:
    auction_lines = prepare_order_lines(auction_raw, metric_mode=metric_mode)
    # Auction 表本身就是用户指定的 Auction 订单范围：
    # 不再套用左侧 Created Time 日期筛选，否则历史 Auction 总量会被截断。
    # 例如文件内共有 284 个 Auction Order，但侧边栏选 2026/05/01–2026/05/11 时只会剩 68。
    if auction_lines.empty:
        return {"has_data": False}

    auction_orders = build_order_level(auction_lines, live1_start, live1_end, live2_start, live2_end)
    auction_ids = set(auction_orders["Order ID"])
    auction_cancel_orders = auction_orders[auction_orders["Is Cancelled"]].copy()
    auction_cancel_lines = auction_lines[auction_lines["Order ID"].isin(set(auction_cancel_orders["Order ID"]))].copy()
    auction_return_orders = auction_orders[auction_orders["Has Return Refund in All Order"]].copy()

    # Prefer uploaded Returned Order table for detailed return metrics and return reason distribution.
    # The matching key is Order ID. Returned table is treated as all returned records and does not use Order Status.
    auction_return_lines = pd.DataFrame()
    return_order_ids_from_return_table = set()
    source = "All Order 表中的 Cancelation/Return Type = Return/Refund"
    if return_lines is not None and not return_lines.empty:
        auction_return_lines = return_lines[return_lines["Order ID"].isin(auction_ids)].copy()
        if not auction_return_lines.empty:
            return_order_ids_from_return_table = set(auction_return_lines["Order ID"].dropna().astype(str))
            source = "Returned Order 表按 Order ID 匹配 Auction 订单"
    if auction_return_lines.empty:
        auction_return_lines = make_auction_return_lines_from_all_order(auction_lines)

    return_metrics = build_return_metrics(auction_return_lines, top_n) if not auction_return_lines.empty else None

    total_orders = len(auction_orders)
    cancel_count = len(auction_cancel_orders)
    # Count/AOV for "申请退货" follows the Auction order table's Return/Refund order flag when available,
    # so the core Auction count stays consistent with the uploaded Auction detail table.
    # Returned Order table is used for detailed reason distribution by Order ID match.
    if not auction_return_orders.empty:
        return_order_ids = set(auction_return_orders["Order ID"].dropna().astype(str))
    elif return_order_ids_from_return_table:
        return_order_ids = return_order_ids_from_return_table
    elif return_metrics:
        return_order_ids = set(auction_return_lines["Order ID"].dropna().astype(str))
    else:
        return_order_ids = set()
    return_count = len(return_order_ids)

    non_cancelled_orders_df = auction_orders[~auction_orders["Is Cancelled"]].copy()
    return_aov_orders_df = non_cancelled_orders_df[non_cancelled_orders_df["Order ID"].isin(return_order_ids)].copy()
    auction_aov_excl_cancelled = mean_order_amount(non_cancelled_orders_df)
    return_aov = mean_order_amount(return_aov_orders_df)
    aov_gap = return_aov - auction_aov_excl_cancelled if not pd.isna(return_aov) and not pd.isna(auction_aov_excl_cancelled) else np.nan
    aov_summary = pd.DataFrame([
        ["有效 Auction 订单数（已排除 Cancelled）", len(non_cancelled_orders_df)],
        ["全部 Auction 平均 AOV（排除 Cancelled）", fmt_money(auction_aov_excl_cancelled)],
        ["提交退货 Auction 订单数", return_count],
        ["提交退货 Auction 平均 AOV", fmt_money(return_aov)],
        ["退货 AOV - 全部有效 Auction AOV", fmt_money(aov_gap)],
    ], columns=["指标", "数值"])

    live_cancel = int(auction_cancel_orders["Is Live by Created Time"].sum()) if cancel_count else 0
    live_total = int(auction_orders["Is Live by Created Time"].sum()) if total_orders else 0
    auction_sku_breakdown = make_breakdown(auction_lines, "Seller SKU Base", top_n=10, metric_col="Metric Units")
    cancel_reason = value_count_table(auction_cancel_orders, "Cancel Reason Clean", cancel_count, top_n)
    cancel_sku = make_breakdown(auction_cancel_lines, "Seller SKU Base", top_n=10, metric_col="Metric Units")
    cancel_product = make_breakdown(auction_cancel_lines, "Product Link / Product Name", top_n=10, metric_col="Metric Units")

    return {
        "has_data": True,
        "source": source,
        "lines": auction_lines,
        "orders": auction_orders,
        "ids": auction_ids,
        "total_orders": total_orders,
        "cancel_orders": cancel_count,
        "return_orders": return_count,
        "cancel_rate": pct(cancel_count, total_orders),
        "return_rate": pct(return_count, total_orders),
        "return_order_ids": return_order_ids,
        "non_cancelled_orders": len(non_cancelled_orders_df),
        "auction_aov_excl_cancelled": auction_aov_excl_cancelled,
        "return_aov": return_aov,
        "aov_gap": aov_gap,
        "aov_summary": aov_summary,
        "return_aov_orders_df": return_aov_orders_df,
        "live_total": live_total,
        "live_cancel": live_cancel,
        "live_cancel_rate": pct(live_cancel, live_total),
        "sku_breakdown": auction_sku_breakdown,
        "cancel_orders_df": auction_cancel_orders,
        "cancel_lines_df": auction_cancel_lines,
        "cancel_reason_df": cancel_reason,
        "cancel_sku_df": cancel_sku,
        "cancel_product_df": cancel_product,
        "return_metrics": return_metrics,
        "return_lines": auction_return_lines,
    }


# ============================================================
# HTML / Excel exporters
# ============================================================
def html_table(df: pd.DataFrame, max_rows=20) -> str:
    if df is None or df.empty:
        return '<div class="empty">暂无数据</div>'
    show = df.head(max_rows).copy()
    return show.to_html(index=False, escape=True, classes="data-table")


def base_html(title: str, subtitle: str, cards: list[tuple[str, str, str]], sections: list[tuple[str, str]]) -> str:
    card_html = "\n".join(
        f'<div class="card"><div class="lbl">{html.escape(lbl)}</div><div class="val">{val}</div><div class="sub">{html.escape(sub)}</div></div>'
        for lbl, val, sub in cards
    )
    section_html = "\n".join(f'<section><h2>{html.escape(h)}</h2>{body}</section>' for h, body in sections)
    return f"""
<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans SC',Arial,sans-serif;background:#f7f7f5;color:#222;margin:0;padding:36px}}
header{{border-bottom:1px solid #ddd;padding-bottom:22px;margin-bottom:28px}}h1{{font-size:34px;margin:0 0 8px}}.subtitle{{color:#666}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.card{{background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:18px}}
.lbl{{font-size:12px;color:#666;text-transform:uppercase;letter-spacing:.06em}}.val{{font-size:32px;font-weight:700;margin:8px 0}}.sub{{font-size:12px;color:#888}}
section{{background:#fff;border:1px solid #e5e5e5;border-radius:12px;padding:22px;margin:18px 0}}h2{{font-size:18px;margin:0 0 14px}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px}}.data-table th,.data-table td{{border-bottom:1px solid #eee;text-align:left;padding:9px 10px;vertical-align:top}}
.data-table th{{background:#fafafa;color:#555}}.empty{{color:#999;font-style:italic}}
</style></head><body><header><h1>{html.escape(title)}</h1><div class="subtitle">{html.escape(subtitle)}</div></header><div class="grid">{card_html}</div>{section_html}</body></html>
"""


def build_cancelled_html(ctx: dict) -> str:
    cards = [
        ("Total Orders", fmt_num(ctx["total_orders"]), "Order ID 去重"),
        ("Cancelled Orders", fmt_num(ctx["cancel_orders"]), f"Cancel Rate {ctx['cancel_rate']:.1f}%"),
        ("Live-attributed Cancelled", fmt_num(ctx["live_cancel"]), "按 Created Time 判断"),
        ("Cancelled SKU Units", fmt_num(ctx["cancel_sku_units"]), ctx["metric_label"]),
    ]
    sections = [
        ("直播归因", html_table(ctx["live_summary"])),
        ("Cancel Reasons", html_table(ctx["reason_df"])),
        ("甲型 / SKU", html_table(ctx["style_breakdown"])),
        ("产品链接", html_table(ctx["product_breakdown"])),
    ]
    return base_html(ctx["title"], ctx["subtitle"], cards, sections)


def build_returned_html(ctx: dict) -> str:
    if not ctx.get("has_return_data"):
        return base_html("Returned Orders Report", "未上传 Returned Order 表", [], [("提示", "请上传 Returned Order 表后生成报告。")])
    rm = ctx["return_metrics"]
    cards = [
        ("Returned Packages", fmt_num(rm["total_packages"]), "按 tracking / return order / order 去重"),
        ("Created in Live Time", fmt_num(rm["live_packages"]), fmt_pct(rm["live_packages"], rm["total_packages"])),
        ("Seller Fault", fmt_num(int(rm["packages"]["Seller Fault Flag"].sum())), "Reason ≠ No Longer Needed"),
        ("Refund Only", fmt_num(int(rm["packages"]["Refund Only Flag"].sum())), "L column Return Type"),
    ]
    sections = [
        ("直播归因", html_table(rm["segment_summary"])),
        ("Returned 核心指标", html_table(rm["boolean_summary"])),
        ("Return Reason", html_table(rm["reason_df"])),
        ("退货 SKU Top10（款式英文名）", html_table(rm["sku_top10"])),
        ("Top5 高退货产品链接", html_table(rm["product_top5"])),
        ("指定产品链接退货占比", html_table(rm["product_targets"])),
    ]
    return base_html(ctx["title"], ctx["subtitle"], cards, sections)


def build_auction_html(ctx: dict) -> str:
    if not ctx.get("has_data"):
        return base_html("Auction Orders Report", "未上传 Auction 订单表", [], [("提示", "请上传 Auction 订单详情表后生成报告。")])
    cards = [
        ("总 Auction Order 数", fmt_num(ctx["total_orders"]), "Order ID 去重"),
        ("Cancelled 订单", fmt_num(ctx["cancel_orders"]), f"Cancel Rate {ctx['cancel_rate']:.1f}%"),
        ("申请退货 Return/Refund", fmt_num(ctx["return_orders"]), f"退货率 {ctx['return_rate']:.1f}%"),
        ("有效 Auction 平均 AOV", fmt_money(ctx.get("auction_aov_excl_cancelled", np.nan)), "已排除 Cancelled"),
        ("退货 Auction 平均 AOV", fmt_money(ctx.get("return_aov", np.nan)), "按 Order ID 匹配 Returned 表"),
        ("直播创建订单", fmt_num(ctx["live_total"]), "按 Created Time 判断"),
    ]
    sections = [
        ("Auction 订单总览", html_table(pd.DataFrame([
            ["总 Order 数", ctx["total_orders"]],
            ["Cancelled 订单", ctx["cancel_orders"]],
            ["申请退货（Return/Refund）", ctx["return_orders"]],
            ["退货率", f"{ctx['return_orders']} / {ctx['total_orders']} = {ctx['return_rate']:.1f}%"],
            ["有效 Auction 订单数（已排除 Cancelled）", ctx.get("non_cancelled_orders", 0)],
            ["全部 Auction 平均 AOV（排除 Cancelled）", fmt_money(ctx.get("auction_aov_excl_cancelled", np.nan))],
            ["提交退货 Auction 平均 AOV", fmt_money(ctx.get("return_aov", np.nan))],
        ], columns=["指标", "数值"]))),
        ("Auction AOV 分析", html_table(ctx.get("aov_summary", pd.DataFrame()))),
        ("Auction Seller SKU 分布", html_table(ctx["sku_breakdown"])),
        ("Auction Cancel Reasons", html_table(ctx["cancel_reason_df"])),
        ("Auction Cancel SKU", html_table(ctx["cancel_sku_df"])),
        ("Auction Cancel 产品链接", html_table(ctx["cancel_product_df"])),
    ]
    if ctx.get("return_metrics"):
        rm = ctx["return_metrics"]
        sections.extend([
            ("Auction Return 来源", f"<p>{html.escape(ctx['source'])}</p>"),
            ("Auction Returned 核心指标", html_table(rm["boolean_summary"])),
            ("Auction Return Reason", html_table(rm["reason_df"])),
            ("Auction Return SKU / 款式", html_table(rm["sku_top10"])),
            ("Auction Return Product Link", html_table(rm["product_top5"])),
        ])
    return base_html(ctx["title"], ctx["subtitle"], cards, sections)


def make_excel_download(cancel_ctx: dict, return_ctx: dict, auction_ctx: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Cancelled sheets
        pd.DataFrame([
            ["Total Orders", cancel_ctx["total_orders"]],
            ["Cancelled Orders", cancel_ctx["cancel_orders"]],
            ["Cancel Rate", f"{cancel_ctx['cancel_rate']:.1f}%"],
        ], columns=["Metric", "Value"]).to_excel(writer, index=False, sheet_name="Cancelled Summary")
        for name, df in [
            ("Cancel Live", cancel_ctx["live_summary"]),
            ("Cancel Reasons", cancel_ctx["reason_df"]),
            ("Cancel SKU", cancel_ctx["style_breakdown"]),
            ("Cancel Products", cancel_ctx["product_breakdown"]),
            ("Cancelled Orders", cancel_ctx["cancel_orders_df"]),
        ]:
            df.to_excel(writer, index=False, sheet_name=name[:31])

        # Returned sheets
        if return_ctx.get("has_return_data"):
            rm = return_ctx["return_metrics"]
            for name, df in [
                ("Return Segment", rm["segment_summary"]),
                ("Return KPI", rm["boolean_summary"]),
                ("Return Reasons", rm["reason_df"]),
                ("Return SKU Top10", rm["sku_top10"]),
                ("Return Product Top5", rm["product_top5"]),
                ("Return Product Targets", rm["product_targets"]),
                ("Return Packages", rm["packages"]),
                ("Return Lines", rm["lines"]),
            ]:
                df.to_excel(writer, index=False, sheet_name=name[:31])

        # Auction sheets
        if auction_ctx.get("has_data"):
            pd.DataFrame([
                ["总 Order 数", auction_ctx["total_orders"]],
                ["Cancelled 订单", auction_ctx["cancel_orders"]],
                ["申请退货（Return/Refund）", auction_ctx["return_orders"]],
                ["退货率", f"{auction_ctx['return_orders']} / {auction_ctx['total_orders']} = {auction_ctx['return_rate']:.1f}%"],
                ["有效 Auction 订单数（已排除 Cancelled）", auction_ctx.get("non_cancelled_orders", 0)],
                ["全部 Auction 平均 AOV（排除 Cancelled）", fmt_money(auction_ctx.get("auction_aov_excl_cancelled", np.nan))],
                ["提交退货 Auction 平均 AOV", fmt_money(auction_ctx.get("return_aov", np.nan))],
                ["退货 AOV - 全部有效 Auction AOV", fmt_money(auction_ctx.get("aov_gap", np.nan))],
            ], columns=["指标", "数值"]).to_excel(writer, index=False, sheet_name="Auction Summary")
            if "aov_summary" in auction_ctx:
                auction_ctx["aov_summary"].to_excel(writer, index=False, sheet_name="Auction AOV")
            for name, df in [
                ("Auction Orders", auction_ctx["orders"]),
                ("Auction SKU", auction_ctx["sku_breakdown"]),
                ("Auction Cancel Reasons", auction_ctx["cancel_reason_df"]),
                ("Auction Cancel SKU", auction_ctx["cancel_sku_df"]),
                ("Auction Cancel Products", auction_ctx["cancel_product_df"]),
            ]:
                df.to_excel(writer, index=False, sheet_name=name[:31])
            if auction_ctx.get("return_metrics"):
                rm = auction_ctx["return_metrics"]
                for name, df in [
                    ("Auction Return KPI", rm["boolean_summary"]),
                    ("Auction Return Reasons", rm["reason_df"]),
                    ("Auction Return SKU", rm["sku_top10"]),
                    ("Auction Return Products", rm["product_top5"]),
                    ("Auction Return Lines", auction_ctx["return_lines"]),
                ]:
                    df.to_excel(writer, index=False, sheet_name=name[:31])

        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#F3F4F6", "border": 1})
        for ws in writer.sheets.values():
            ws.freeze_panes(1, 0)
            ws.set_row(0, None, header_fmt)
            ws.set_column(0, 0, 28)
            ws.set_column(1, 20, 18)
    return output.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.title("💅 Cancelled + Returned + Auction Orders Report Generator")
st.success("✅ 版本确认：AUCTION_AOV_RETURN_REASON_20260511｜Auction 上传入口已启用；Auction 不受左侧日期区间截断；Auction AOV 已使用 Z column Order Amount")
st.caption("上传订单总表生成 Cancelled 报告；可额外上传 Returned Order 表和 Auction 订单表，分别生成独立分析板块。")

with st.expander("核心口径说明", expanded=True):
    st.markdown(
        """
- **Cancelled**：看订单总表 B column `Order Status`，`Canceled/Cancelled` 视作取消；一个 `Order ID` 只算一个 cancel。
- **Returned**：Returned Order 表上传后，全部视作 returned，不看 Order Status；按包裹去重，优先用 `Return Logistics Tracking ID`，缺失时用 `Return Order ID / Order ID`。
- **直播归因**：用 `Created Time` 判断是否属于直播时间，不用 `Cancelled Time` 判断直播归因。
- **Auction**：上传 Auction 订单详情表后，程序按 `Order ID` 去重，统计总 Auction Order、Cancelled、Return/Refund、退货率，并单独分析 Auction 的 cancel / return 指标。
- **Auction Return/Refund**：优先用 Returned Order 表按 Order ID 匹配 Auction 订单做详细退货原因分析；如果没上传 Returned Order 表，则用 Auction 表里的 `Cancelation/Return Type = Return/Refund` 做申请退货统计。
- **Auction AOV**：使用 Z column `Order Amount`，按 Order ID 去重后计算；全部 Auction 平均 AOV 会排除 Cancelled 订单，提交退货平均 AOV 用 Auction Order ID 对齐 Returned Order 表。
        """
    )

all_order_file = st.file_uploader("1）上传订单总表 CSV / Excel（必传）", type=["csv", "xlsx", "xls"], key="all_order")
return_file = st.file_uploader("2）可选：上传 Returned Order 表 CSV / Excel", type=["csv", "xlsx", "xls"], key="return_order")
catalog_file = st.file_uploader("3）可选：上传产品图册 CSV / Excel（用于退货 SKU 匹配款式英文名）", type=["csv", "xlsx", "xls"], key="catalog")
auction_file = st.file_uploader("4）【NEW】可选：上传 Auction 订单详情表 CSV / Excel", type=["csv", "xlsx", "xls"], key="auction_order")

if all_order_file is None:
    st.info("请先上传订单总表。")
    st.stop()

try:
    raw_all = clean_df_columns(read_uploaded_file(all_order_file))
except Exception as e:
    st.error(f"订单总表读取失败：{e}")
    st.stop()

missing = validate_all_order_columns(raw_all)
if missing:
    st.error("订单总表缺少必要字段：" + ", ".join(missing))
    st.write("当前识别到的字段：", list(raw_all.columns))
    st.stop()

with st.sidebar:
    st.header("Report Settings")
    metric_mode_label = st.radio("SKU / 产品链接统计口径", ["按 Quantity 汇总（推荐）", "按 SKU 行数汇总"], index=0)
    metric_mode = "quantity" if metric_mode_label.startswith("按 Quantity") else "rows"
    live1_start = st.number_input("直播①开始", 0, 23, 10, 1)
    live1_end = st.number_input("直播①结束", 1, 24, 18, 1)
    live2_start = st.number_input("直播②开始", 0, 23, 19, 1)
    live2_end = st.number_input("直播②结束", 1, 24, 23, 1)
    top_n = st.slider("Breakdown 显示 Top N", 5, 30, 10, 1)

if not (live1_start < live1_end and live2_start < live2_end):
    st.error("直播开始时间必须小于结束时间。")
    st.stop()

all_lines = prepare_order_lines(raw_all, metric_mode)
valid_created = all_lines[all_lines["Created Datetime"].notna()].copy()
if valid_created.empty:
    st.error("Created Time 无法解析，无法继续。")
    st.stop()

min_date = valid_created["Created Datetime"].dt.date.min()
max_date = valid_created["Created Datetime"].dt.date.max()
with st.sidebar:
    date_range = st.date_input("选择 Created Time 日期区间", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

selected_lines = filter_by_created_date(all_lines, start_date, end_date)
if selected_lines.empty:
    st.warning("当前日期区间内没有订单。")
    st.stop()

all_orders = build_order_level(selected_lines, live1_start, live1_end, live2_start, live2_end)
cancel_orders = all_orders[all_orders["Is Cancelled"]].copy()
cancel_lines = selected_lines[selected_lines["Order ID"].isin(set(cancel_orders["Order ID"]))].copy()

metric_label = "按 Quantity 汇总" if metric_mode == "quantity" else "按 SKU 行数汇总"
style_breakdown = make_breakdown(cancel_lines, "Nail Style", top_n, metric_col="Metric Units")
product_breakdown = make_breakdown(cancel_lines, "Product Link / Product Name", top_n, metric_col="Metric Units")
reason_df = value_count_table(cancel_orders, "Cancel Reason Clean", len(cancel_orders), top_n)

live1_cancel = int((cancel_orders["Live Segment by Created Time"] == "直播①").sum()) if not cancel_orders.empty else 0
live2_cancel = int((cancel_orders["Live Segment by Created Time"] == "直播②").sum()) if not cancel_orders.empty else 0
live_cancel = live1_cancel + live2_cancel
nonlive_cancel = len(cancel_orders) - live_cancel
live1_all = int((all_orders["Live Segment by Created Time"] == "直播①").sum()) if not all_orders.empty else 0
live2_all = int((all_orders["Live Segment by Created Time"] == "直播②").sum()) if not all_orders.empty else 0
live_all = live1_all + live2_all
nonlive_all = len(all_orders) - live_all
live_summary = pd.DataFrame([
    ["直播①", live1_cancel, live1_all, pct(live1_cancel, live1_all), fmt_pct(live1_cancel, len(cancel_orders))],
    ["直播②", live2_cancel, live2_all, pct(live2_cancel, live2_all), fmt_pct(live2_cancel, len(cancel_orders))],
    ["直播合计", live_cancel, live_all, pct(live_cancel, live_all), fmt_pct(live_cancel, len(cancel_orders))],
    ["非直播", nonlive_cancel, nonlive_all, pct(nonlive_cancel, nonlive_all), fmt_pct(nonlive_cancel, len(cancel_orders))],
], columns=["Segment", "Cancelled Orders", "Total Created Orders in Segment", "Segment Cancel Rate", "% of Cancelled Orders"])

cancel_ctx = {
    "title": f"{start_date}–{end_date} Cancelled Orders Report",
    "subtitle": f"Source: {all_order_file.name} · Order ID 去重",
    "total_orders": len(all_orders),
    "cancel_orders": len(cancel_orders),
    "cancel_rate": pct(len(cancel_orders), len(all_orders)),
    "live_cancel": live_cancel,
    "cancel_sku_units": float(cancel_lines["Metric Units"].sum()) if not cancel_lines.empty else 0,
    "metric_label": metric_label,
    "live_summary": live_summary,
    "reason_df": reason_df,
    "style_breakdown": style_breakdown,
    "product_breakdown": product_breakdown,
    "cancel_orders_df": cancel_orders,
    "cancel_lines_df": cancel_lines,
}

# Returned analysis.
catalog_map = {}
if catalog_file is not None:
    try:
        catalog_map = build_catalog_sku_name_map(read_uploaded_file(catalog_file))
    except Exception as e:
        st.warning(f"产品图册读取失败，退货 SKU 将使用 J column 兜底：{e}")

return_ctx = {"has_return_data": False, "title": f"{start_date}–{end_date} Returned Orders Report", "subtitle": "未上传 Returned Order 表"}
return_lines = pd.DataFrame()
if return_file is not None:
    try:
        raw_return = read_uploaded_file(return_file)
        return_lines = prepare_return_lines(raw_return, all_orders[["Order ID", "Created Datetime"]], catalog_map, live1_start, live1_end, live2_start, live2_end)
        # Keep same Created Time range as the main report when matched created time exists.
        if not return_lines.empty and "Created Datetime" in return_lines.columns:
            has_dt = return_lines["Created Datetime"].notna()
            in_range = return_lines["Created Datetime"].dt.date.between(start_date, end_date)
            return_lines = return_lines[~has_dt | in_range].copy()
        return_metrics = build_return_metrics(return_lines, top_n)
        return_ctx = {
            "has_return_data": True,
            "title": f"{start_date}–{end_date} Returned Orders Report",
            "subtitle": f"Source: {return_file.name} · Returned 表全部视作 returned",
            "return_metrics": return_metrics,
        }
    except Exception as e:
        st.warning(f"Returned Order 表读取/分析失败：{e}")

# Auction analysis.
auction_ctx = {"has_data": False, "title": f"{start_date}–{end_date} Auction Orders Report", "subtitle": "未上传 Auction 订单表"}
if auction_file is not None:
    try:
        raw_auction = read_uploaded_file(auction_file)
        auction_ctx = build_auction_metrics(
            raw_auction, selected_lines, return_lines, metric_mode,
            live1_start, live1_end, live2_start, live2_end,
            start_date, end_date, top_n,
        )
        auction_ctx["title"] = f"{start_date}–{end_date} Auction Orders Report"
        auction_ctx["subtitle"] = f"Source: {auction_file.name} · Auction Seller SKU 通常为 1/2/3"
    except Exception as e:
        st.warning(f"Auction 订单表读取/分析失败：{e}")

cancelled_html = build_cancelled_html(cancel_ctx)
returned_html = build_returned_html(return_ctx)
auction_html = build_auction_html(auction_ctx)
excel_bytes = make_excel_download(cancel_ctx, return_ctx, auction_ctx)

# ============================================================
# Display tabs
# ============================================================
main_tabs = st.tabs(["Cancelled Report", "Returned Report", "Auction Report", "Downloads"])

with main_tabs[0]:
    st.subheader("Cancelled Orders 核心结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders", fmt_num(cancel_ctx["total_orders"]))
    c2.metric("Cancelled Orders", fmt_num(cancel_ctx["cancel_orders"]))
    c3.metric("Cancel Rate", f"{cancel_ctx['cancel_rate']:.1f}%")
    c4.metric("Live-attributed Cancelled", fmt_num(cancel_ctx["live_cancel"]))
    st.dataframe(live_summary, use_container_width=True, hide_index=True)
    t1, t2, t3, t4 = st.tabs(["Cancel Reasons", "甲型 / SKU", "产品链接", "订单级 Cancelled 明细"])
    with t1:
        st.dataframe(reason_df, use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(style_breakdown, use_container_width=True, hide_index=True)
    with t3:
        st.dataframe(product_breakdown, use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(cancel_orders, use_container_width=True, hide_index=True)
    components.html(cancelled_html, height=900, scrolling=True)

with main_tabs[1]:
    if not return_ctx.get("has_return_data"):
        st.info("上传 Returned Order 表后，这里会显示 Returned Report。")
    else:
        rm = return_ctx["return_metrics"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Returned Packages", fmt_num(rm["total_packages"]))
        r2.metric("Created in Live Time", fmt_num(rm["live_packages"]), fmt_pct(rm["live_packages"], rm["total_packages"]))
        r3.metric("Seller Fault", fmt_num(int(rm["packages"]["Seller Fault Flag"].sum())))
        r4.metric("Refund Only", fmt_num(int(rm["packages"]["Refund Only Flag"].sum())))
        st.dataframe(rm["segment_summary"], use_container_width=True, hide_index=True)
        rt = st.tabs(["核心指标", "Return Reason", "退货 SKU Top10", "Top5 产品链接", "指定产品链接占比", "Returned 明细"])
        with rt[0]:
            st.dataframe(rm["boolean_summary"], use_container_width=True, hide_index=True)
        with rt[1]:
            st.dataframe(rm["reason_df"], use_container_width=True, hide_index=True)
        with rt[2]:
            st.caption("H column Seller SKU 去掉尺码后缀，并通过产品图册匹配款式英文名；只展示 Top10。")
            st.dataframe(rm["sku_top10"], use_container_width=True, hide_index=True)
        with rt[3]:
            st.caption("I column Product Name / 产品链接 Top5。")
            st.dataframe(rm["product_top5"], use_container_width=True, hide_index=True)
        with rt[4]:
            st.dataframe(rm["product_targets"], use_container_width=True, hide_index=True)
        with rt[5]:
            st.dataframe(rm["packages"], use_container_width=True, hide_index=True)
        components.html(returned_html, height=900, scrolling=True)

with main_tabs[2]:
    if not auction_ctx.get("has_data"):
        st.info("上传 Auction 订单详情表后，这里会显示 Auction Report。")
    else:
        st.subheader("Auction 订单分析")
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("总 Auction Order 数", fmt_num(auction_ctx["total_orders"]))
        a2.metric("Cancelled 订单", fmt_num(auction_ctx["cancel_orders"]), f"{auction_ctx['cancel_rate']:.1f}%")
        a3.metric("申请退货", fmt_num(auction_ctx["return_orders"]), f"退货率 {auction_ctx['return_rate']:.1f}%")
        a4.metric("有效 Auction 平均 AOV", fmt_money(auction_ctx.get("auction_aov_excl_cancelled", np.nan)))
        a5.metric("退货 Auction 平均 AOV", fmt_money(auction_ctx.get("return_aov", np.nan)))
        auction_overview = pd.DataFrame([
            ["总 Order 数", auction_ctx["total_orders"]],
            ["Cancelled 订单", auction_ctx["cancel_orders"]],
            ["申请退货（Return/Refund）", auction_ctx["return_orders"]],
            ["退货率", f"{auction_ctx['return_orders']} / {auction_ctx['total_orders']} = {auction_ctx['return_rate']:.1f}%"],
            ["有效 Auction 订单数（已排除 Cancelled）", auction_ctx.get("non_cancelled_orders", 0)],
            ["全部 Auction 平均 AOV（排除 Cancelled）", fmt_money(auction_ctx.get("auction_aov_excl_cancelled", np.nan))],
            ["提交退货 Auction 平均 AOV", fmt_money(auction_ctx.get("return_aov", np.nan))],
            ["退货 AOV - 全部有效 Auction AOV", fmt_money(auction_ctx.get("aov_gap", np.nan))],
        ], columns=["指标", "数值"])
        st.dataframe(auction_overview, use_container_width=True, hide_index=True)
        at = st.tabs(["Auction SKU 分布", "Auction Cancel", "Auction Return", "Auction AOV", "Auction 明细"])
        with at[0]:
            st.caption("G column Seller SKU 通常为 1 / 2 / 3。")
            st.dataframe(auction_ctx["sku_breakdown"], use_container_width=True, hide_index=True)
        with at[1]:
            st.dataframe(auction_ctx["cancel_reason_df"], use_container_width=True, hide_index=True)
            st.dataframe(auction_ctx["cancel_sku_df"], use_container_width=True, hide_index=True)
            st.dataframe(auction_ctx["cancel_product_df"], use_container_width=True, hide_index=True)
        with at[2]:
            st.caption(f"Return 数据来源：{auction_ctx.get('source', '-')}")
            if auction_ctx.get("return_metrics"):
                arm = auction_ctx["return_metrics"]
                st.dataframe(arm["boolean_summary"], use_container_width=True, hide_index=True)
                st.dataframe(arm["reason_df"], use_container_width=True, hide_index=True)
                st.dataframe(arm["sku_top10"], use_container_width=True, hide_index=True)
                st.dataframe(arm["product_top5"], use_container_width=True, hide_index=True)
            else:
                st.info("未识别到 Auction Return/Refund 数据。")
        with at[3]:
            st.caption("AOV 使用 Z column Order Amount；全部 Auction 平均 AOV 已排除 Cancelled 订单。")
            st.dataframe(auction_ctx.get("aov_summary", pd.DataFrame()), use_container_width=True, hide_index=True)
            if not auction_ctx.get("return_aov_orders_df", pd.DataFrame()).empty:
                st.caption("提交退货 Auction 订单明细（用于计算退货平均 AOV）")
                st.dataframe(auction_ctx["return_aov_orders_df"], use_container_width=True, hide_index=True)
        with at[4]:
            st.dataframe(auction_ctx["orders"], use_container_width=True, hide_index=True)
        components.html(auction_html, height=900, scrolling=True)

with main_tabs[3]:
    st.subheader("下载报告")
    file_stub = f"order_reports_{start_date}_{end_date}"
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("下载 Cancelled HTML", cancelled_html.encode("utf-8"), f"cancelled_report_{start_date}_{end_date}.html", "text/html", use_container_width=True)
    with d2:
        st.download_button("下载 Returned HTML", returned_html.encode("utf-8"), f"returned_report_{start_date}_{end_date}.html", "text/html", use_container_width=True)
    with d3:
        st.download_button("下载 Auction HTML", auction_html.encode("utf-8"), f"auction_report_{start_date}_{end_date}.html", "text/html", use_container_width=True)
    with d4:
        st.download_button("下载全部 Excel", excel_bytes, f"{file_stub}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.caption("说明：直播归因使用 Created Time；Cancel 峰值/Return 申请行为与直播因果关系仍需结合直播 GMV、订单量、主播话术和活动价变动一起判断。")
