import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "COLLECTION_OPTIMIZED_20260515_PASTEABLE"
CANCELLED = {"cancelled", "canceled"}
SIZE_SUFFIX_RE = re.compile(r"[-_ ]?(XS|S|M|L|XL|XXL|XXXL)$", re.I)

st.set_page_config(
    page_title="Cancelled + Returned + Auction Orders Report Generator",
    page_icon="💅",
    layout="wide",
)

COLLECTION_PATTERNS = [
    (["dreamwear", "dream wear"], "DreamWear Collection", "达人带货"),
    (["top trend"], "TOP TREND Collection", "达人带货"),
    (["buy 4 get 1 free", "buy 4"], "Buy 4 Get 1 Free", "官号视频"),
    (["next gen", "nextgen"], "Next Gen Collection", "直播间"),
    (["square"], "Square Collection", "直播间"),
    (["secret", "final sale"], "Secret + Final Sale Collection", "直播间"),
    (["almond"], "Almond (Shape) Collection", "直播间"),
    (["stiletto"], "Stiletto Collection", "直播间"),
    (["new drop"], "NEW DROP Collection", "直播间"),
    (["spring bloom"], "SPRING BLOOM Collection", "直播间"),
    (["summer shine"], "SUMMER SHINE Collection", "直播间"),
    (["valentine"], "Valentine Collection", "直播间"),
    (["christmas glow", "christmas"], "Christmas Glow Collection", "直播间"),
    (["best seller", "bestseller"], "Best Seller Collection", "直播间"),
    (["organizer binder"], "Organizer Binder", "直播间"),
    (["toolkit", "toolkits", "10 pcs"], "10 PCs TOOLKITS", "直播间"),
]

TARGET_LINKS = [
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


def norm(x, default=""):
    if pd.isna(x):
        return default
    s = str(x).replace("\u3000", " ").strip()
    return s if s else default


def clean_cols(df):
    df = df.copy()
    df.columns = [norm(c).replace("\ufeff", "") for c in df.columns]
    return df


def read_file(f):
    if f is None:
        return None
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return clean_cols(pd.read_csv(f, dtype=str, encoding="utf-8-sig"))
        except UnicodeDecodeError:
            f.seek(0)
            return clean_cols(pd.read_csv(f, dtype=str, encoding="latin1"))
    return clean_cols(pd.read_excel(f, dtype=str))


def col(df, *names, idx=None):
    low = {c.lower().strip(): c for c in df.columns}

    for n in names:
        k = n.lower().strip()
        if k in low:
            return low[k]

    for n in names:
        k = n.lower().strip()
        for c in df.columns:
            if k in c.lower().strip():
                return c

    if idx is not None and idx < len(df.columns):
        return df.columns[idx]

    return None


def as_id(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[$,% ,]", "", regex=True),
        errors="coerce",
    )


def to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def pct(n, d):
    return 0 if not d else n / d * 100


def fmt_money(x):
    return "$0.00" if pd.isna(x) else f"${x:,.2f}"


def remove_size(s):
    return SIZE_SUFFIX_RE.sub("", norm(s)).strip()


def collection_name(x):
    raw = norm(x, "Unknown")
    low = raw.lower()

    for keys, label, _channel in COLLECTION_PATTERNS:
        if any(k in low for k in keys):
            return label

    m = re.search(r"([^|–—-]{1,80}?collection)", raw, flags=re.I)
    if m:
        return m.group(1).strip()

    return raw[:90]


def collection_channel(name):
    low = norm(name).lower()

    for keys, _label, channel in COLLECTION_PATTERNS:
        if any(k in low for k in keys):
            return channel

    if "collection" in low:
        return "直播间"

    return "其他"


def is_cancelled(s):
    return s.astype(str).str.strip().str.lower().isin(CANCELLED)


def is_live_time(dt, start1, end1, start2, end2):
    if pd.isna(dt):
        return np.nan
    h = dt.hour
    return (start1 <= h <= end1) or (start2 <= h <= end2)


def order_level(df):
    order_col = col(df, "Order ID", idx=0)
    if order_col is None:
        raise ValueError("找不到 Order ID column")

    d = df.copy()
    d[order_col] = as_id(d[order_col])

    status = col(d, "Order Status", idx=1)
    created = col(d, "Created Time", idx=27)
    cancelled_time = col(d, "Cancelled Time", idx=32)
    cancel_reason = col(d, "Cancel Reason", idx=34)
    order_amount = col(d, "Order Amount", idx=25)
    ret_type = col(
        d,
        "Cancelation/Return Type",
        "Cancellation/Return Type",
        idx=3,
    )

    agg = {status: "first"} if status else {}

    for c in [created, cancelled_time, cancel_reason, order_amount, ret_type]:
        if c and c not in agg:
            agg[c] = "first"

    od = d.groupby(order_col, dropna=False).agg(agg).reset_index()

    if created:
        od[created] = to_dt(od[created])
    if cancelled_time:
        od[cancelled_time] = to_dt(od[cancelled_time])
    if order_amount:
        od[order_amount] = to_num(od[order_amount])

    return od, {
        "order": order_col,
        "status": status,
        "created": created,
        "cancelled_time": cancelled_time,
        "cancel_reason": cancel_reason,
        "order_amount": order_amount,
        "return_type": ret_type,
    }


def count_table(df, label_col, value_col=None, denom=None, name="Item", top=None):
    if df is None or df.empty or label_col is None:
        return pd.DataFrame(columns=[name, "数量", "占比 %"])

    base = df.copy()
    base[label_col] = base[label_col].map(lambda x: norm(x, "Unknown"))

    if value_col and value_col in base.columns:
        g = base.groupby(label_col)[value_col].sum(min_count=1).reset_index(name="数量")
    else:
        g = base.groupby(label_col).size().reset_index(name="数量")

    denom = denom if denom is not None else g["数量"].sum()
    g["占比 %"] = g["数量"].apply(lambda x: pct(x, denom))
    g = g.sort_values("数量", ascending=False)

    if top:
        g = g.head(top)

    return g.rename(columns={label_col: name})


def reason_table(df, reason_col, denom=None, name="Reason"):
    return count_table(df, reason_col, denom=denom, name=name)


def build_catalog_map(cat):
    if cat is None or cat.empty:
        return {}

    sku = col(cat, "SKU", idx=0)
    eng = col(cat, "款式英文名称", "English", "Style", idx=2)

    if not sku or not eng:
        return {}

    m = {}

    for _, r in cat[[sku, eng]].dropna(how="all").iterrows():
        key = remove_size(r[sku]).upper()
        val = norm(r[eng])
        if key and val:
            m[key] = val

    return m


def cancelled_context(all_df, start1, end1, start2, end2, metric_mode):
    order_df, oc = order_level(all_df)

    total_orders = order_df[oc["order"]].nunique()

    cancel_order_ids = (
        set(order_df.loc[is_cancelled(order_df[oc["status"]]), oc["order"]])
        if oc["status"]
        else set()
    )

    cancel_orders = len(cancel_order_ids)
    order_col = oc["order"]

    lines = all_df.copy()
    lines[order_col] = as_id(lines[order_col])
    cancel_lines = lines[lines[order_col].isin(cancel_order_ids)].copy()

    product_col = col(lines, "Product Name", idx=7)
    sku_col = col(lines, "Seller SKU", idx=6)
    qty_col = col(lines, "Quantity", idx=10)
    reason_col = oc["cancel_reason"]
    amount_col = oc["order_amount"]
    created_col = oc["created"]
    cancelled_time_col = oc["cancelled_time"]

    if qty_col:
        cancel_lines["__qty"] = to_num(cancel_lines[qty_col]).fillna(1)
    else:
        cancel_lines["__qty"] = 1

    if sku_col:
        cancel_lines["__sku_base"] = cancel_lines[sku_col].map(remove_size)

    if product_col:
        cancel_lines["__collection"] = cancel_lines[product_col].map(collection_name)

    metric = "__qty" if metric_mode == "Quantity" else None

    sku_breakdown = (
        count_table(
            cancel_lines,
            "__sku_base",
            metric,
            denom=cancel_lines[metric].sum() if metric else len(cancel_lines),
            name="甲型 / SKU",
            top=20,
        )
        if sku_col
        else pd.DataFrame()
    )

    product_breakdown = (
        count_table(
            cancel_lines,
            product_col,
            metric,
            denom=cancel_lines[metric].sum() if metric else len(cancel_lines),
            name="产品链接",
            top=20,
        )
        if product_col
        else pd.DataFrame()
    )

    reason_df = (
        reason_table(
            order_df[order_df[order_col].isin(cancel_order_ids)],
            reason_col,
            denom=cancel_orders,
            name="Cancel Reason",
        )
        if reason_col
        else pd.DataFrame()
    )

    collection_df = pd.DataFrame()

    if product_col and cancel_orders:
        tmp = cancel_lines[[order_col, product_col]].copy()
        tmp["Collection"] = tmp[product_col].map(collection_name)
        tmp = tmp[
            tmp["Collection"].str.contains(
                "Collection|BUY 4|TOOLKITS|Organizer",
                case=False,
                na=False,
            )
        ]

        collection_df = (
            tmp.groupby("Collection")[order_col]
            .nunique()
            .reset_index(name="Cancelled 行数")
        )

        collection_df["Cancelled 占比"] = collection_df["Cancelled 行数"].apply(
            lambda x: pct(x, cancel_orders)
        )
        collection_df["类型"] = collection_df["Collection"].map(collection_channel)

        collection_df = collection_df[
            ["Collection", "类型", "Cancelled 行数", "Cancelled 占比"]
        ].sort_values("Cancelled 行数", ascending=False)

    live_summary = pd.DataFrame()

    if created_col:
        tmp = order_df[order_df[order_col].isin(cancel_order_ids)].copy()
        tmp["__live"] = tmp[created_col].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )

        rows = [
            [
                "直播①",
                int(
                    (
                        (tmp[created_col].dt.hour >= start1)
                        & (tmp[created_col].dt.hour <= end1)
                    ).sum()
                ),
            ],
            [
                "直播②",
                int(
                    (
                        (tmp[created_col].dt.hour >= start2)
                        & (tmp[created_col].dt.hour <= end2)
                    ).sum()
                ),
            ],
            ["直播合计", int((tmp["__live"] == True).sum())],
            ["非直播", int((tmp["__live"] == False).sum())],
            ["Unknown", int(tmp["__live"].isna().sum())],
        ]

        live_summary = pd.DataFrame(rows, columns=["Segment", "Cancelled Orders"])
        live_summary["占比 %"] = live_summary["Cancelled Orders"].apply(
            lambda x: pct(x, cancel_orders)
        )

    return {
        "total_orders": total_orders,
        "cancel_orders": cancel_orders,
        "cancel_rate": pct(cancel_orders, total_orders),
        "order_df": order_df,
        "cancel_lines": cancel_lines,
        "order_col": order_col,
        "sku_breakdown": sku_breakdown,
        "product_breakdown": product_breakdown,
        "reason_df": reason_df,
        "collection_df": collection_df,
        "live_summary": live_summary,
        "amount_col": amount_col,
        "created_col": created_col,
        "cancelled_time_col": cancelled_time_col,
    }


def returned_context(ret_df, all_ctx, cat_map, start1, end1, start2, end2, metric_mode):
    d = ret_df.copy()

    order_col = col(d, "Order ID", idx=1)
    return_order_col = col(d, "Return Order ID", idx=0)
    track_col = col(d, "Return Logistics Tracking ID", idx=16)
    sku_col = col(d, "Seller SKU", idx=7)
    product_col = col(d, "Product Name", idx=8)
    sku_name_col = col(d, "SKU Name", idx=9)
    return_type_col = col(d, "Return Type", idx=11)
    reason_col = col(d, "Return Reason", idx=13)
    qty_col = col(d, "Return Quantity", idx=15)
    sub_col = col(d, "Return Sub Status", idx=18)

    if order_col:
        d[order_col] = as_id(d[order_col])

    d["__qty"] = to_num(d[qty_col]).fillna(1) if qty_col else 1

    if sku_col:
        d["__sku_base"] = d[sku_col].map(remove_size).str.upper()
    else:
        d["__sku_base"] = "Unknown"

    if sku_name_col:
        fallback_style = d[sku_name_col].map(remove_size)
    else:
        fallback_style = d["__sku_base"]

    d["__style"] = d["__sku_base"].map(cat_map).fillna(fallback_style)

    if product_col:
        d["__collection"] = d[product_col].map(collection_name)
    else:
        d["__collection"] = "Unknown"

    keys = []

    for _, r in d.iterrows():
        key = norm(r.get(track_col)) if track_col else ""

        if not key and return_order_col:
            key = norm(r.get(return_order_col))

        if not key and order_col:
            key = norm(r.get(order_col))

        keys.append(key or f"row_{len(keys)}")

    d["__package_key"] = keys
    pkg = d.drop_duplicates("__package_key").copy()
    returned_packages = len(pkg)

    if order_col and all_ctx.get("created_col"):
        od = all_ctx["order_df"]
        created_map = od.set_index(all_ctx["order_col"])[all_ctx["created_col"]]
        pkg["__created"] = pkg[order_col].map(created_map)
        pkg["__live"] = pkg["__created"].apply(
            lambda x: is_live_time(x, start1, end1, start2, end2)
        )
    else:
        pkg["__live"] = np.nan

    live_created = int((pkg["__live"] == True).sum())

    metric = "__qty" if metric_mode == "Quantity" else None

    sku_top10 = count_table(
        d,
        "__style",
        metric,
        denom=d[metric].sum() if metric else len(d),
        name="款式英文名",
        top=10,
    )

    reason_df = (
        reason_table(d, reason_col, denom=len(d), name="Return Reason")
        if reason_col
        else pd.DataFrame()
    )

    product_top5 = (
        count_table(
            d,
            product_col,
            metric,
            denom=d[metric].sum() if metric else len(d),
            name="产品链接",
            top=5,
        )
        if product_col
        else pd.DataFrame()
    )

    target_rows = []

    if product_col:
        low = d[product_col].astype(str).str.lower()

        for key in TARGET_LINKS:
            mask = low.str.contains(re.escape(key.lower()), na=False)
            n = int(d.loc[mask, metric].sum()) if metric else int(mask.sum())
            target_rows.append(
                [
                    key,
                    n,
                    pct(n, d[metric].sum() if metric else len(d)),
                ]
            )

    target_link_df = pd.DataFrame(
        target_rows,
        columns=["Product Link Keyword", "退货行数", "占比 %"],
    )

    collection_df = pd.DataFrame()

    if product_col:
        tmp = d.copy()
        tmp["Collection"] = tmp[product_col].map(collection_name)

        raw_has_collection = tmp[product_col].astype(str).str.contains(
            "Collection|BUY 4|TOOLKIT|Organizer",
            case=False,
            na=False,
        )

        tmp = tmp[raw_has_collection]

        collection_df = tmp.groupby("Collection").size().reset_index(name="退货行数")
        collection_df["退货占比"] = collection_df["退货行数"].apply(
            lambda x: pct(x, len(d))
        )
        collection_df["类型"] = collection_df["Collection"].map(collection_channel)

        collection_df = collection_df[
            ["Collection", "类型", "退货行数", "退货占比"]
        ].sort_values("退货行数", ascending=False)

    seller_fault_n = 0

    if reason_col:
        seller_fault_n = int(
            (d[reason_col].astype(str).str.strip().str.lower() != "no longer needed").sum()
        )

    request_cancelled_n = (
        int(d[sub_col].astype(str).str.contains("cancel", case=False, na=False).sum())
        if sub_col
        else 0
    )

    shipped_back_n = int(d[track_col].map(norm).ne("").sum()) if track_col else 0

    refund_only_n = (
        int(
            d[return_type_col]
            .astype(str)
            .str.contains("refund only", case=False, na=False)
            .sum()
        )
        if return_type_col
        else 0
    )

    return {
        "lines": d,
        "package_df": pkg,
        "returned_packages": returned_packages,
        "live_created": live_created,
        "live_pct": pct(live_created, returned_packages),
        "unknown_created": int(pkg["__live"].isna().sum()),
        "sku_top10": sku_top10,
        "reason_df": reason_df,
        "product_top5": product_top5,
        "target_link_df": target_link_df,
        "collection_df": collection_df,
        "seller_fault_n": seller_fault_n,
        "seller_fault_pct": pct(seller_fault_n, len(d)),
        "request_cancelled_n": request_cancelled_n,
        "request_cancelled_pct": pct(request_cancelled_n, len(d)),
        "shipped_back_n": shipped_back_n,
        "shipped_back_pct": pct(shipped_back_n, len(d)),
        "refund_only_n": refund_only_n,
        "refund_only_pct": pct(refund_only_n, len(d)),
        "order_col": order_col,
        "reason_col": reason_col,
    }


def merge_collection_summary(ret_ctx, can_ctx):
    ret = ret_ctx.get("collection_df", pd.DataFrame()) if ret_ctx else pd.DataFrame()
    can = can_ctx.get("collection_df", pd.DataFrame()) if can_ctx else pd.DataFrame()

    names = sorted(set(ret.get("Collection", [])) | set(can.get("Collection", [])))

    rows = []

    for name in names:
        r = ret[ret["Collection"] == name]
        c = can[can["Collection"] == name]
        channel = collection_channel(name)

        rows.append(
            [
                name,
                channel,
                int(r["退货行数"].iloc[0]) if not r.empty else 0,
                float(r["退货占比"].iloc[0]) if not r.empty else 0,
                int(c["Cancelled 行数"].iloc[0]) if not c.empty else 0,
                float(c["Cancelled 占比"].iloc[0]) if not c.empty else 0,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "链接名称",
            "类型",
            "退货行数",
            "退货占比",
            "Cancelled",
            "Cancel占比",
        ],
    )

    if df.empty:
        return df, pd.DataFrame()

    df["差值 pp"] = df["Cancel占比"] - df["退货占比"]
    df = df.sort_values(["退货行数", "Cancelled"], ascending=False)

    channel_rows = []

    for ch in ["达人带货", "官号视频", "直播间"]:
        sub = df[df["类型"] == ch]
        if not sub.empty:
            channel_rows.append(
                [
                    f"{ch}小计",
                    "",
                    sub["退货行数"].sum(),
                    sub["退货占比"].sum(),
                    sub["Cancelled"].sum(),
                    sub["Cancel占比"].sum(),
                ]
            )

    channel_rows.append(
        [
            "全部 Collection 合计",
            "",
            df["退货行数"].sum(),
            df["退货占比"].sum(),
            df["Cancelled"].sum(),
            df["Cancel占比"].sum(),
        ]
    )

    summary = pd.DataFrame(
        channel_rows,
        columns=[
            "链接名称",
            "类型",
            "退货行数",
            "退货占比",
            "Cancelled",
            "Cancel占比",
        ],
    )

    return df, summary


def collection_insights(comp, summary):
    if comp is None or comp.empty:
        return ["暂无 Collection 数据。"]

    top_ret = comp.sort_values("退货行数", ascending=False).iloc[0]
    top_cancel = comp.sort_values("Cancelled", ascending=False).iloc[0]

    live = (
        summary[summary["链接名称"].str.contains("直播间", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    creator = (
        summary[summary["链接名称"].str.contains("达人", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    official = (
        summary[summary["链接名称"].str.contains("官号", na=False)]
        if summary is not None and not summary.empty
        else pd.DataFrame()
    )

    notes = []

    notes.append(
        f"1. **{top_ret['链接名称']}** 是退货贡献最高的链接，退货占比 {top_ret['退货占比']:.2f}%，"
        f"Cancelled 占比 {top_ret['Cancel占比']:.2f}%。建议结合该链接的 Return Reason 和高退货 SKU 单独深挖。"
    )

    notes.append(
        f"2. **{top_cancel['链接名称']}** 是 Cancelled 贡献最高的链接，Cancelled 占比 {top_cancel['Cancel占比']:.2f}%，"
        f"退货占比 {top_cancel['退货占比']:.2f}%。如果 Cancel 明显高于退货，优先排查履约时效、备货、价格敏感或预期落差。"
    )

    if not creator.empty:
        r, c = creator["退货占比"].iloc[0], creator["Cancel占比"].iloc[0]
        notes.append(
            f"3. **达人带货链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "达人链路更容易出现内容展示与实物预期差，建议审查达人展示内容、买家评价与款式质量一致性。"
        )

    if not official.empty:
        r, c = official["退货占比"].iloc[0], official["Cancel占比"].iloc[0]
        notes.append(
            f"4. **官号视频链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "若 Cancel 高于退货，说明用户更多在收货前取消，建议优化促销承诺、发货时效与视频引导。"
        )

    if not live.empty:
        r, c = live["退货占比"].iloc[0], live["Cancel占比"].iloc[0]
        notes.append(
            f"5. **直播间链接** 退货占比约 {r:.2f}%，Cancel 占比约 {c:.2f}%。"
            "直播间更偏冲动购买，建议加强尺码引导、材质展示、佩戴效果解释，降低 No Longer Needed 和预期差。"
        )

    return notes


def auction_context(auc_df, ret_ctx):
    order_df, oc = order_level(auc_df)

    order_col = oc["order"]
    status_col = oc["status"]
    ret_type_col = oc["return_type"]
    amount_col = oc["order_amount"]

    total = order_df[order_col].nunique()

    cancelled_ids = (
        set(order_df.loc[is_cancelled(order_df[status_col]), order_col])
        if status_col
        else set()
    )
    cancelled_n = len(cancelled_ids)

    return_ids = set()

    if ret_type_col:
        return_ids |= set(
            order_df.loc[
                order_df[ret_type_col]
                .astype(str)
                .str.contains("return/refund", case=False, na=False),
                order_col,
            ]
        )

    if ret_ctx and ret_ctx.get("order_col"):
        ret_order_col = ret_ctx["order_col"]
        return_ids |= set(ret_ctx["lines"][ret_order_col].dropna().astype(str)) & set(
            order_df[order_col].astype(str)
        )

    return_n = len(return_ids)

    valid = order_df[~order_df[order_col].isin(cancelled_ids)].copy()
    valid_aov = valid[amount_col].mean() if amount_col else np.nan

    ret_valid = order_df[order_df[order_col].isin(return_ids - cancelled_ids)].copy()
    ret_aov = ret_valid[amount_col].mean() if amount_col else np.nan

    lines = auc_df.copy()
    lines[order_col] = as_id(lines[order_col])

    sku_col = col(lines, "Seller SKU", idx=6)
    qty_col = col(lines, "Quantity", idx=10)

    if qty_col:
        lines["__qty"] = to_num(lines[qty_col]).fillna(1)
    else:
        lines["__qty"] = 1

    if sku_col:
        lines["__sku_base"] = lines[sku_col].map(remove_size)

    sku_dist = (
        count_table(
            lines,
            "__sku_base",
            "__qty",
            denom=lines["__qty"].sum(),
            name="Seller SKU Base",
        )
        if sku_col
        else pd.DataFrame()
    )

    auction_return_reason = pd.DataFrame()

    if ret_ctx and ret_ctx.get("order_col") and ret_ctx.get("reason_col"):
        rd = ret_ctx["lines"]
        matched = rd[rd[ret_ctx["order_col"]].isin(set(order_df[order_col]))]
        auction_return_reason = reason_table(
            matched,
            ret_ctx["reason_col"],
            denom=len(matched),
            name="Return Reason",
        )

    cancel_reason = (
        reason_table(
            order_df[order_df[order_col].isin(cancelled_ids)],
            oc["cancel_reason"],
            denom=cancelled_n,
            name="Cancel Reason",
        )
        if oc["cancel_reason"]
        else pd.DataFrame()
    )

    return {
        "total": total,
        "cancelled_n": cancelled_n,
        "return_n": return_n,
        "return_rate": pct(return_n, total),
        "valid_aov": valid_aov,
        "return_aov": ret_aov,
        "aov_diff": ret_aov - valid_aov
        if pd.notna(ret_aov) and pd.notna(valid_aov)
        else np.nan,
        "sku_dist": sku_dist,
        "return_reason": auction_return_reason,
        "cancel_reason": cancel_reason,
    }


def show_metric_row(items):
    cols = st.columns(len(items))

    for c, (label, value, delta) in zip(cols, items):
        c.metric(label, value, delta=delta)


def style_pct_df(df, pct_cols=None):
    if df is None or df.empty:
        return df

    out = df.copy()

    pct_cols = pct_cols or [
        c
        for c in out.columns
        if "占比" in c or c.endswith("%") or "Cancel占比" in c or "退货占比" in c
    ]

    for c in pct_cols:
        if c in out.columns:
            out[c] = out[c].apply(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float, np.floating)) else x
            )

    return out


def excel_bytes(sheets):
    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets:
            safe = re.sub(r"[\\/*?:\[\]]", " ", name)[:31]
            (df if df is not None else pd.DataFrame()).to_excel(
                writer,
                sheet_name=safe,
                index=False,
            )

    return bio.getvalue()


st.title("💅 Cancelled + Returned + Auction Orders Report Generator")
st.caption("上传 TikTok Shop 订单总表；可额外上传 Returned Order 表、产品图册、Auction 订单表。")
st.success(
    f"✅ 版本确认：{APP_VERSION}｜含 Auction + AOV + Collection 渠道汇总表 + 链路启示分析"
)

with st.expander("这版程序的核心口径", expanded=True):
    st.markdown(
        """
- **Cancelled**：订单总表 B column `Order Status` = `Cancelled/Canceled`；一个 `Order ID` 只算一个 Cancel。
- **Returned**：Returned Order 表上传后全部视为 returned；表是一行一个 SKU。
- **Returned Collection**：读 Returned 表 I column `Product Name`，带 Collection/Buy4/Toolkit/Organizer 的链接按 SKU 行统计，分母是 Returned Excel 总行数。
- **Cancelled Collection**：读 cancelled 对应 SKU 行的 H/Product Name，按 Collection 归类后统计唯一 `Order ID`，分母是总 cancelled orders。
- **直播归因**：用订单总表 AB column `Created Time` 判断，不用 Cancelled Time。
- **Auction**：Auction 表单独统计全表，不受左侧日期筛选；AOV 用 Z column `Order Amount`，排除 cancelled 后计算有效 AOV。
"""
    )

with st.sidebar:
    st.header("Report Settings")

    metric_mode = st.radio(
        "SKU / 产品链接统计口径",
        ["Quantity", "SKU 行数"],
        index=0,
        horizontal=False,
    )

    st.caption("Quantity = 用 Quantity / Return Quantity 汇总；SKU 行数 = 一行算一条。")

    st.subheader("直播时间")

    c1, c2 = st.columns(2)
    live1_start = c1.number_input("直播①开始", 0, 23, 10)
    live1_end = c2.number_input("直播①结束", 0, 23, 18)

    c3, c4 = st.columns(2)
    live2_start = c3.number_input("直播②开始", 0, 23, 19)
    live2_end = c4.number_input("直播②结束", 0, 23, 23)

    top_n = st.slider("Breakdown 显示 Top N", 5, 30, 10)


all_file = st.file_uploader(
    "1）上传订单总表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="all_order",
)

returned_file = st.file_uploader(
    "2）可选：上传 Returned Order 表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="returned_order",
)

product_catalog_file = st.file_uploader(
    "3）可选：上传产品图册 CSV / Excel（用于 Seller SKU 匹配款式英文名）",
    type=["csv", "xlsx", "xls"],
    key="catalog",
)

auction_file = st.file_uploader(
    "4）【NEW】可选：上传 Auction 订单详情表 CSV / Excel",
    type=["csv", "xlsx", "xls"],
    key="auction_file",
)

if not all_file:
    st.info("请先上传 TikTok Shop 导出的订单总表。")
    st.stop()


all_df = read_file(all_file)
cat_df = read_file(product_catalog_file) if product_catalog_file else None
ret_df = read_file(returned_file) if returned_file else None
auction_df = read_file(auction_file) if auction_file else None

cat_map = build_catalog_map(cat_df)

try:
    cancel_ctx = cancelled_context(
        all_df,
        live1_start,
        live1_end,
        live2_start,
        live2_end,
        metric_mode,
    )

    return_ctx = (
        returned_context(
            ret_df,
            cancel_ctx,
            cat_map,
            live1_start,
            live1_end,
            live2_start,
            live2_end,
            metric_mode,
        )
        if ret_df is not None
        else None
    )

    comp_df, channel_summary = merge_collection_summary(return_ctx, cancel_ctx)
    insights = collection_insights(comp_df, channel_summary)

    auction_ctx = (
        auction_context(auction_df, return_ctx)
        if auction_df is not None
        else None
    )

except Exception as e:
    st.error(f"程序读取失败：{e}")
    st.stop()


st.markdown("## 报告输出")

tab_names = ["Cancelled Report"]

if return_ctx:
    tab_names.append("Returned Report")

if auction_ctx:
    tab_names.append("Auction Report")

tab_names.append("Collection 汇总")
tab_names.append("Downloads")

tabs = st.tabs(tab_names)

idx = 0

with tabs[idx]:
    st.header("Cancelled Orders 核心结果")

    show_metric_row(
        [
            ("总 Order 数", f"{cancel_ctx['total_orders']:,}", None),
            (
                "Cancelled 订单",
                f"{cancel_ctx['cancel_orders']:,}",
                f"{cancel_ctx['cancel_rate']:.2f}%",
            ),
            ("Cancelled Rate", f"{cancel_ctx['cancel_rate']:.2f}%", None),
        ]
    )

    sub = st.tabs(
        [
            "Cancel Reasons",
            "甲型 / SKU",
            "产品链接",
            "Collection 链接",
            "直播归因",
        ]
    )

    with sub[0]:
        st.dataframe(
            style_pct_df(cancel_ctx["reason_df"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[1]:
        st.dataframe(
            style_pct_df(cancel_ctx["sku_breakdown"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[2]:
        st.dataframe(
            style_pct_df(cancel_ctx["product_breakdown"].head(top_n)),
            use_container_width=True,
            hide_index=True,
        )

    with sub[3]:
        st.dataframe(
            style_pct_df(cancel_ctx["collection_df"]),
            use_container_width=True,
            hide_index=True,
        )

    with sub[4]:
        st.dataframe(
            style_pct_df(cancel_ctx["live_summary"]),
            use_container_width=True,
            hide_index=True,
        )

idx += 1


if return_ctx:
    with tabs[idx]:
        st.header("Returned Orders 核心结果")

        show_metric_row(
            [
                ("Returned Packages", f"{return_ctx['returned_packages']:,}", None),
                (
                    "Created in Live Time",
                    f"{return_ctx['live_created']:,}",
                    f"{return_ctx['live_pct']:.2f}%",
                ),
                (
                    "Unknown Created Time",
                    f"{return_ctx['unknown_created']:,}",
                    None,
                ),
                (
                    "Seller Fault",
                    f"{return_ctx['seller_fault_n']:,}",
                    f"{return_ctx['seller_fault_pct']:.2f}%",
                ),
            ]
        )

        show_metric_row(
            [
                (
                    "Request Cancelled",
                    f"{return_ctx['request_cancelled_n']:,}",
                    f"{return_ctx['request_cancelled_pct']:.2f}%",
                ),
                (
                    "已寄出退回包裹",
                    f"{return_ctx['shipped_back_n']:,}",
                    f"{return_ctx['shipped_back_pct']:.2f}%",
                ),
                (
                    "Refund Only",
                    f"{return_ctx['refund_only_n']:,}",
                    f"{return_ctx['refund_only_pct']:.2f}%",
                ),
            ]
        )

        sub = st.tabs(
            [
                "Return Reasons",
                "Top10 退货款式",
                "Top5 产品链接",
                "指定链接占比",
                "Collection 链接",
            ]
        )

        with sub[0]:
            st.dataframe(
                style_pct_df(return_ctx["reason_df"].head(top_n)),
                use_container_width=True,
                hide_index=True,
            )

        with sub[1]:
            st.dataframe(
                style_pct_df(return_ctx["sku_top10"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[2]:
            st.dataframe(
                style_pct_df(return_ctx["product_top5"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[3]:
            st.dataframe(
                style_pct_df(return_ctx["target_link_df"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[4]:
            st.dataframe(
                style_pct_df(return_ctx["collection_df"]),
                use_container_width=True,
                hide_index=True,
            )

    idx += 1


if auction_ctx:
    with tabs[idx]:
        st.header("Auction 订单分析")

        show_metric_row(
            [
                ("总 Auction Order 数", f"{auction_ctx['total']:,}", None),
                (
                    "Cancelled 订单",
                    f"{auction_ctx['cancelled_n']:,}",
                    f"{pct(auction_ctx['cancelled_n'], auction_ctx['total']):.2f}%",
                ),
                (
                    "申请退货（Return/Refund）",
                    f"{auction_ctx['return_n']:,}",
                    f"退货率 {auction_ctx['return_rate']:.2f}%",
                ),
                (
                    "有效 Auction 平均 AOV",
                    fmt_money(auction_ctx["valid_aov"]),
                    "排除 Cancelled",
                ),
            ]
        )

        show_metric_row(
            [
                (
                    "提交退货 Auction 平均 AOV",
                    fmt_money(auction_ctx["return_aov"]),
                    None,
                ),
                (
                    "退货 AOV - 有效 AOV",
                    fmt_money(auction_ctx["aov_diff"]),
                    None,
                ),
            ]
        )

        st.table(
            pd.DataFrame(
                [
                    ["总 Order 数", auction_ctx["total"]],
                    ["Cancelled 订单", auction_ctx["cancelled_n"]],
                    ["申请退货（Return/Refund）", auction_ctx["return_n"]],
                    [
                        "退货率",
                        f"{auction_ctx['return_n']} / {auction_ctx['total']} = {auction_ctx['return_rate']:.2f}%",
                    ],
                    [
                        "有效 Auction 平均 AOV（排除 Cancelled）",
                        fmt_money(auction_ctx["valid_aov"]),
                    ],
                    [
                        "提交退货 Auction 平均 AOV",
                        fmt_money(auction_ctx["return_aov"]),
                    ],
                ],
                columns=["指标", "数值"],
            )
        )

        sub = st.tabs(
            [
                "Auction SKU 分布",
                "Auction Cancel",
                "Auction Return Reason",
            ]
        )

        with sub[0]:
            st.dataframe(
                style_pct_df(auction_ctx["sku_dist"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[1]:
            st.dataframe(
                style_pct_df(auction_ctx["cancel_reason"]),
                use_container_width=True,
                hide_index=True,
            )

        with sub[2]:
            st.dataframe(
                style_pct_df(auction_ctx["return_reason"]),
                use_container_width=True,
                hide_index=True,
            )

    idx += 1


with tabs[idx]:
    st.header("Collection 链接综合分析")

    st.markdown("#### 明细表")
    st.dataframe(
        style_pct_df(comp_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 汇总表（按渠道类型）")
    st.dataframe(
        style_pct_df(channel_summary),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 启示")

    for note in insights:
        st.markdown(note)

    action_df = pd.DataFrame(
        [
            [
                "达人带货",
                "收货后退货率高 → 内容/实物预期差",
                "审查达人展示内容，强化真实买家评价和细节展示",
            ],
            [
                "官号视频",
                "Cancel 率高 → 下单后反悔 / 等待耐心差",
                "优化 Buy 4 促销设计、发货时效和视频承诺",
            ],
            [
                "直播间",
                "冲动购买结构性问题",
                "加强尺码选择、材质展示和售前提醒，减少 No Longer Needed",
            ],
        ],
        columns=["渠道", "核心问题", "建议方向"],
    )

    st.markdown("#### 渠道动作建议表")
    st.dataframe(
        action_df,
        use_container_width=True,
        hide_index=True,
    )

idx += 1


with tabs[idx]:
    st.header("Downloads")

    sheets = [
        ("Cancel Reasons", cancel_ctx.get("reason_df")),
        ("Cancel SKU", cancel_ctx.get("sku_breakdown")),
        ("Cancel Product Links", cancel_ctx.get("product_breakdown")),
        ("Cancel Collection Links", cancel_ctx.get("collection_df")),
        ("Cancel Live", cancel_ctx.get("live_summary")),
        ("Collection Comparison", comp_df),
        ("Collection Channel Summary", channel_summary),
        ("Channel Action", action_df),
    ]

    if return_ctx:
        sheets.extend(
            [
                ("Return Reasons", return_ctx.get("reason_df")),
                ("Return SKU Top10", return_ctx.get("sku_top10")),
                ("Return Product Top5", return_ctx.get("product_top5")),
                ("Return Target Links", return_ctx.get("target_link_df")),
                ("Return Collection Links", return_ctx.get("collection_df")),
            ]
        )

    if auction_ctx:
        sheets.extend(
            [
                ("Auction SKU", auction_ctx.get("sku_dist")),
                ("Auction Cancel", auction_ctx.get("cancel_reason")),
                ("Auction Return Reason", auction_ctx.get("return_reason")),
            ]
        )

    st.download_button(
        "下载 Excel Report",
        data=excel_bytes(sheets),
        file_name="cancel_return_auction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
