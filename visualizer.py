import plotly.graph_objects as go
from datetime import datetime
from dash import Dash, html, Output, Input, callback, State, dcc, ALL, ctx
import dash

from datastream.yfinance_ohlcv import YFinanceOHLCVFetcher as Fetcher
from datastream.finnhub_ohlcv import api_key, FinnhubOHLCVFetcher as newsFetcher
from XeniaV2 import create_default_system
from sp500 import sp500_tickers


fetcher = Fetcher()

# ── PALETTE ─────────────────────────────────────────────
AMBER = '#F39F41'
BLACK = '#000000'
DARK  = '#0a0a0a'
DIM   = '#222222'
GREY  = '#555555'
WHITE = '#FFFFFF'
GREEN = '#00FF41'
RED   = '#FF3B3B'

# ── STYLES ──────────────────────────────────────────────
page_style = dict(
    backgroundColor=BLACK, fontFamily='Courier New', color=WHITE,
    height='100vh', margin='0', padding='0',
    overflow='hidden', boxSizing='border-box',
    display='flex', flexDirection='column'
)
header_style = dict(
    display='flex', alignItems='center', justifyContent='space-between',
    backgroundColor=DARK, borderBottom=f'2px solid {AMBER}',
    padding='0.6vh 1vw', flexShrink=0, gap='1vw'
)
chart_panel_header_style = dict(
    color=AMBER, fontSize='0.6vw', fontWeight='bold', letterSpacing='3px',
    padding='0.4vh 1vw', borderBottom=f'1px solid {AMBER}',
    backgroundColor=DARK, flexShrink=0,
    display='flex', justifyContent='space-between', alignItems='center'
)
section_label_style = dict(
    color=AMBER, fontSize='0.6vw', fontWeight='bold', letterSpacing='3px',
    padding='0.4vh 1vw', borderBottom=f'1px solid {AMBER}',
    backgroundColor=DARK, flexShrink=0,
    display='flex', justifyContent='space-between', alignItems='center'
)
chart_container_style = dict(display='flex', flexGrow=1, overflow='hidden')
chart_panel_style     = dict(display='flex', flexDirection='column', width='75%',
                              borderRight=f'1px solid {AMBER}', overflow='hidden')
sidebar_style         = dict(display='flex', flexDirection='column', width='25%',
                              overflow='hidden', backgroundColor=DARK)
timeframe_bar_style   = dict(display='flex', alignItems='center', gap='1.5vw',
                              backgroundColor=DARK, borderTop=f'1px solid {AMBER}',
                              padding='0.5vh 1vw', flexShrink=0)
news_ticker_style     = dict(backgroundColor='#050505', borderTop=f'1px solid {DIM}',
                              padding='0.4vh 0', flexShrink=0,
                              overflow='hidden', whiteSpace='nowrap')
chart_style           = dict(flexGrow=1, overflow='hidden', display='flex')
timer_style           = dict(fontSize='0.9vw', fontWeight='bold', color=AMBER,
                              fontFamily='Courier New', letterSpacing='2px')
signal_h_style        = dict(fontFamily='Courier New', fontWeight='bold',
                              fontSize='0.95vw', color=AMBER, letterSpacing='2px')
news_panel_style      = dict(height='15vh', borderTop=f'1px solid {AMBER}',
                              backgroundColor='#050505', flexShrink=0,
                              overflow='hidden', display='flex', flexDirection='column')

TICKER_NEWS = (
    "MARKETS  ──  S&P 500 FUTURES STEADY AS FED MINUTES LOOM  ──  "
    "TECH LEADS GAINS IN EARLY TRADING  ──  OIL NEAR 3-MONTH HIGH  ──  "
    "YIELDS EDGE LOWER AHEAD OF JOBS DATA  ──  DOLLAR FLAT ON RATE SHIFT  ──  "
    "GOLD HITS 6-WEEK HIGH ON SAFE-HAVEN DEMAND  ──  BTC CONSOLIDATES ABOVE KEY SUPPORT  ──  "
)

def _news(symbol, lookback=2, line_len=5):
    newsfetcher = newsFetcher(api_key)

    return newsfetcher.headline_line(
        symbol=symbol,
        lookback=lookback,
        line_len=line_len
    )

# ── BUTTON HELPERS ───────────────────────────────────────
def _btn(active=False):
    return dict(
        backgroundColor=AMBER if active else 'transparent',
        border=f'1px solid {AMBER if active else GREY}',
        color=BLACK if active else GREY,
        fontFamily='Courier New', fontSize='0.6vw',
        cursor='pointer', padding='0.2vh 0.6vw', letterSpacing='1px'
    )

def _tab_btn(active=False):
    return dict(
        backgroundColor=AMBER if active else 'transparent',
        border='none', borderBottom=f'2px solid {AMBER if active else "transparent"}',
        color=AMBER if active else GREY,
        fontFamily='Courier New', fontSize='0.6vw',
        cursor='pointer', padding='0.6vh 1vw', letterSpacing='2px',
        fontWeight='bold', flexGrow=1
    )

def _input_style(width='100%'):
    return dict(
        backgroundColor='#111', border=f'1px solid {DIM}',
        color=AMBER, fontFamily='Courier New', fontSize='0.7vw',
        padding='0.4vh 0.5vw', width=width, boxSizing='border-box',
        outline='none'
    )

def _trade_label(text):
    return html.Div(text, style=dict(
        color=GREY, fontSize='0.55vw', letterSpacing='2px', marginBottom='0.3vh'
    ))

# ── MT5 HELPERS ──────────────────────────────────────────
def _mt5_init():
    try:
        import MetaTrader5 as mt5
        return mt5.initialize(), mt5
    except ImportError:
        return False, None

def _mt5_account():
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            return None
        return dict(
            balance=info.balance, equity=info.equity,
            margin=info.margin,   free=info.margin_free,
            profit=info.profit,   name=info.name,
            server=info.server,   leverage=info.leverage
        )
    except Exception:
        return None


def _get_filling_type(symbol):
    """
    Resolve the correct ORDER_FILLING_* constant for a symbol.
    Fixes ERR 10030 — symbol_info.filling_mode is a bitmask,
    not the constant to pass directly in the request.

    Bitmask:  bit 0 = FOK supported,  bit 1 = IOC supported
    Constants: ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1, ORDER_FILLING_RETURN=2
    """
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC  # safe fallback
        filling = info.filling_mode  # bitmask
        if filling & 1:
            return mt5.ORDER_FILLING_FOK  # bit 0 set
        elif filling & 2:
            return mt5.ORDER_FILLING_IOC  # bit 1 set
        else:
            return mt5.ORDER_FILLING_RETURN  # neither → RETURN
    except Exception:
        return 1

def _mt5_positions():
    try:
        import MetaTrader5 as mt5
        pos = mt5.positions_get()
        if not pos:
            return []
        return [dict(
            ticket=p.ticket, symbol=p.symbol,
            volume=p.volume, type='BUY' if p.type == 0 else 'SELL',
            open_price=p.price_open, current=p.price_current,
            sl=p.sl, tp=p.tp, profit=p.profit, comment=p.comment
        ) for p in pos]
    except Exception:
        return []

def _mt5_orders():
    try:
        import MetaTrader5 as mt5
        orders = mt5.orders_get()
        if not orders:
            return []
        type_map = {0:'BUY', 1:'SELL', 2:'BUY LMT', 3:'SELL LMT',
                    4:'BUY STP', 5:'SELL STP'}
        return [dict(
            ticket=o.ticket, symbol=o.symbol,
            volume=o.volume_initial,
            type=type_map.get(o.type, '?'),
            price=o.price_open, sl=o.sl, tp=o.tp
        ) for o in orders]
    except Exception:
        return []


def _mt5_send(symbol, direction, volume, sl=0.0, tp=0.0,
              order_type='market', limit_price=0.0):
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False, f'Symbol {symbol} not found in MT5'

        is_buy = direction == 'buy'
        type_fill = _get_filling_type(symbol)  # ← ERR 10030 fix

        if order_type == 'market':
            action = mt5.TRADE_ACTION_DEAL
            exec_price = tick.ask if is_buy else tick.bid
            order_t = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        elif order_type == 'limit':
            action = mt5.TRADE_ACTION_PENDING
            exec_price = float(limit_price)
            order_t = mt5.ORDER_TYPE_BUY_LIMIT if is_buy else mt5.ORDER_TYPE_SELL_LIMIT
        else:  # stop
            action = mt5.TRADE_ACTION_PENDING
            exec_price = float(limit_price)
            order_t = mt5.ORDER_TYPE_BUY_STOP if is_buy else mt5.ORDER_TYPE_SELL_STOP

        request = {
            "action": action,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_t,
            "price": exec_price,
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "magic": 234000,
            "comment": "XENIA TERMINAL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_fill,  # ← ERR 10030 fix
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, f'OK  #{res.order}  {direction.upper()} {volume} {symbol}'
        return False, f'ERR {res.retcode}: {res.comment}'
    except Exception as e:
        return False, str(e)


def _mt5_close(ticket):
    try:
        import MetaTrader5 as mt5
        pos = mt5.positions_get(ticket=int(ticket))
        if not pos:
            return False, 'Position not found'
        p = pos[0]
        tick = mt5.symbol_info_tick(p.symbol)
        close_t = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        close_px = tick.bid if p.type == 0 else tick.ask
        type_fill = _get_filling_type(p.symbol)  # ← ERR 10030 fix

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_t,
            "position": p.ticket,
            "price": close_px,
            "magic": 234000,
            "comment": "XENIA close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": type_fill,  # ← ERR 10030 fix
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, f'CLOSED #{ticket}'
        return False, f'ERR {res.retcode}: {res.comment} -----am here ----'
    except Exception as e:
        return False, str(e)


def _mt5_close_all():
    results = []
    for p in _mt5_positions():
        ok, msg = _mt5_close(p['ticket'])
        results.append(msg)
    return results


def _mt5_cancel_order(ticket):
    try:
        import MetaTrader5 as mt5
        res = mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE,
                              "order": int(ticket)})
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True, f'CANCELLED #{ticket}'
        return False, f'ERR {res.retcode}: {res.comment}'
    except Exception as e:
        return False, str(e)


# ── RENDER HELPERS ───────────────────────────────────────
def _base_fig():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=BLACK, paper_bgcolor=BLACK,
        font=dict(color=AMBER, family='Courier New', size=11),
        margin=dict(l=55, r=15, t=35, b=40),
        autosize=True, showlegend=False,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor=DIM, color=AMBER,
                   showline=True, linecolor=AMBER, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor=DIM, color=AMBER,
                   showline=True, linecolor=AMBER, zeroline=False,
                   tickfont=dict(size=10), side='right'),
        modebar=dict(bgcolor=BLACK, color=GREY, activecolor=AMBER)
    )
    return fig

def _graph(fig):
    return dcc.Graph(
        figure=fig, style={'height': '100%', 'width': '100%'},
        config=dict(displayModeBar=True, displaylogo=False,
                    modeBarButtonsToRemove=['select2d', 'lasso2d', 'autoScale2d'],
                    responsive=True)
    )

def _signal_placeholder():
    return html.Div([
        html.Div("──────────────", style=dict(color=DIM, fontSize='0.7vw', marginBottom='1vh')),
        html.Div("SELECT TICKER AND SEARCH",
                 style=dict(color=GREY, fontSize='0.65vw', letterSpacing='2px')),
    ], style=dict(padding='1.5vh 1.2vw'))

def _signal_panel(ticker, signal, confidence, recmd):
    sig   = recmd.upper()
    color = GREEN if 'BUY' in sig else RED if 'SELL' in sig else AMBER
    try:    conf_str = f'{float(confidence):.1%}'
    except: conf_str = str(confidence)

    def row(label, val, c=AMBER, sz='1.1vw'):
        return html.Div([
            html.Div(label, style=dict(color=GREY, fontSize='0.55vw',
                     letterSpacing='2px', marginBottom='0.2vh')),
            html.Div(val,   style=dict(color=c, fontSize=sz,
                     fontWeight='bold', marginBottom='1.2vh', letterSpacing='1px'))
        ])
    return html.Div([
        row("TICKER",     ticker,   AMBER, '1.2vw'),
        row("SIGNAL",     sig,      color, '1.4vw'),
        row("CONFIDENCE", conf_str, AMBER, '1vw'),
        html.Hr(style=dict(borderColor=DIM, margin='0.6vh 0')),
        html.Div(datetime.now().strftime('%Y-%m-%d %H:%M'),
                 style=dict(color=GREY, fontSize='0.55vw')),
    ], style=dict(padding='1.5vh 1.2vw'))

def _watchlist_row(ticker):
    return html.Div([
        html.Div(ticker,
            id={'type': 'watchlist-item', 'index': ticker},
            style=dict(color=AMBER, fontSize='0.75vw', fontWeight='bold',
                       cursor='pointer', flexGrow=1, letterSpacing='1px')),
        html.Button('×',
            id={'type': 'remove-btn', 'index': ticker},
            style=dict(backgroundColor='transparent', border='none',
                       color=GREY, cursor='pointer', fontSize='0.9vw', padding='0 0.3vw'))
    ], style=dict(display='flex', alignItems='center', justifyContent='space-between',
                  padding='0.5vh 1vw', borderBottom=f'1px solid {DIM}'))

def _acct_row(label, val, color=AMBER):
    return html.Div([
        html.Span(label, style=dict(color=GREY, fontSize='0.55vw',
                  letterSpacing='1px', display='inline-block', width='45%')),
        html.Span(val,   style=dict(color=color, fontSize='0.65vw', fontWeight='bold'))
    ], style=dict(marginBottom='0.4vh'))

def _render_account(info):
    if not info:
        return html.Div("NO DATA", style=dict(color=GREY, fontSize='0.6vw'))
    pnl_color = GREEN if info['profit'] >= 0 else RED
    return html.Div([
        html.Div(f"{info['name']}  ·  {info['server']}",
                 style=dict(color=GREY, fontSize='0.55vw', marginBottom='0.8vh')),
        html.Div([
            html.Div([
                _acct_row("BALANCE",  f"${info['balance']:,.2f}"),
                _acct_row("EQUITY",   f"${info['equity']:,.2f}"),
            ], style=dict(width='50%')),
            html.Div([
                _acct_row("MARGIN",   f"${info['margin']:,.2f}"),
                _acct_row("FREE MRG", f"${info['free']:,.2f}"),
            ], style=dict(width='50%')),
        ], style=dict(display='flex')),
        html.Div([
            html.Span("OPEN P&L  ", style=dict(color=GREY, fontSize='0.55vw')),
            html.Span(f"${info['profit']:+,.2f}",
                      style=dict(color=pnl_color, fontWeight='bold', fontSize='0.75vw')),
            html.Span(f"  1:{info['leverage']}",
                      style=dict(color=GREY, fontSize='0.55vw', marginLeft='0.5vw')),
        ])
    ], style=dict(padding='0.8vh 1vw', borderBottom=f'1px solid {DIM}',
                  backgroundColor='#0d0d0d'))

def _render_positions(positions):
    if not positions:
        return html.Div("NO OPEN POSITIONS",
                        style=dict(color=GREY, fontSize='0.6vw',
                                   letterSpacing='2px', padding='1vh 1vw'))
    rows = []
    for p in positions:
        pnl_color = GREEN if p['profit'] >= 0 else RED
        t_color   = GREEN if p['type'] == 'BUY' else RED
        rows.append(html.Div([
            html.Div([
                html.Span(p['symbol'], style=dict(color=AMBER, fontSize='0.65vw',
                          fontWeight='bold', display='block')),
                html.Span(f"{p['type']}  {p['volume']} lot",
                          style=dict(color=t_color, fontSize='0.55vw')),
            ], style=dict(flexGrow=1)),
            html.Div([
                html.Span(f"${p['profit']:+,.2f}",
                          style=dict(color=pnl_color, fontSize='0.65vw',
                                     fontWeight='bold', display='block')),
                html.Span(f"@ {p['open_price']:.4f}",
                          style=dict(color=GREY, fontSize='0.5vw')),
            ], style=dict(textAlign='right', marginRight='0.5vw')),
            html.Button('×',
                id={'type': 'close-pos', 'index': p['ticket']},
                style=dict(backgroundColor='transparent', border=f'1px solid {RED}',
                           color=RED, cursor='pointer', fontSize='0.7vw',
                           padding='0.2vh 0.4vw', flexShrink=0))
        ], style=dict(display='flex', alignItems='center', padding='0.5vh 0.8vw',
                      borderBottom=f'1px solid {DIM}', gap='0.3vw')))
    return html.Div(rows)

def _render_pending(orders):
    if not orders:
        return html.Div("NO PENDING ORDERS",
                        style=dict(color=GREY, fontSize='0.6vw',
                                   letterSpacing='2px', padding='0.5vh 1vw'))
    rows = []
    for o in orders:
        t_color = GREEN if 'BUY' in o['type'] else RED
        rows.append(html.Div([
            html.Div([
                html.Span(f"{o['symbol']} {o['type']}",
                          style=dict(color=t_color, fontSize='0.6vw',
                                     fontWeight='bold', display='block')),
                html.Span(f"{o['volume']} lot @ {o['price']:.4f}",
                          style=dict(color=GREY, fontSize='0.5vw')),
            ], style=dict(flexGrow=1)),
            html.Button('×',
                id={'type': 'cancel-order', 'index': o['ticket']},
                style=dict(backgroundColor='transparent', border=f'1px solid {GREY}',
                           color=GREY, cursor='pointer', fontSize='0.7vw',
                           padding='0.2vh 0.4vw'))
        ], style=dict(display='flex', alignItems='center', padding='0.4vh 0.8vw',
                      borderBottom=f'1px solid {DIM}', gap='0.3vw')))
    return html.Div(rows)

# ── ORDER FORM ───────────────────────────────────────────
def _order_form():
    import MetaTrader5 as mt5
    mt5.initialize()
    return html.Div([
        # Symbol + Lots
        html.Div([
            html.Div([
                _trade_label("SYMBOL"),
                dcc.Dropdown(options=[symbol.name for symbol in mt5.symbols_get()], id='mt5-symbol')
            ], style=dict(flex=2)),
            html.Div([
                _trade_label("LOTS"),
                dcc.Input(id='mt5-lots', type='number', value=0.01,
                          min=0.01, step=0.01, style=_input_style())
            ], style=dict(flex=1)),
        ], style=dict(display='flex', gap='0.4vw', marginBottom='0.8vh')),

        # Order type
        html.Div([
            _trade_label("ORDER TYPE"),
            dcc.RadioItems(
                id='mt5-order-type',
                options=[{'label': 'MKT', 'value': 'market'},
                         {'label': 'LMT', 'value': 'limit'},
                         {'label': 'STP', 'value': 'stop'}],
                value='market',
                style=dict(display='flex', gap='0.8vw', marginBottom='0.6vh'),
                labelStyle=dict(fontFamily='Courier New', color=AMBER,
                                fontSize='0.65vw', letterSpacing='1px'),
                inputStyle=dict(accentColor=AMBER, cursor='pointer')
            )
        ], style=dict(marginBottom='0.6vh')),

        # Price (limit/stop only)
        html.Div([
            _trade_label("PRICE"),
            dcc.Input(id='mt5-price', type='number', placeholder='0.00',
                      step=0.0001, style=_input_style())
        ], id='mt5-price-row', style=dict(marginBottom='0.6vh', display='none')),

        # SL + TP
        html.Div([
            html.Div([
                _trade_label("STOP LOSS"),
                dcc.Input(id='mt5-sl', type='number', placeholder='0.00',
                          step=0.0001, style=_input_style())
            ], style=dict(flex=1)),
            html.Div([
                _trade_label("TAKE PROFIT"),
                dcc.Input(id='mt5-tp', type='number', placeholder='0.00',
                          step=0.0001, style=_input_style())
            ], style=dict(flex=1)),
        ], style=dict(display='flex', gap='0.4vw', marginBottom='1vh')),

        # BUY / SELL
        html.Div([
            html.Button('▲  BUY', id='mt5-buy', n_clicks=0, style=dict(
                backgroundColor=GREEN, border='none', color=BLACK,
                fontFamily='Courier New', fontWeight='bold',
                fontSize='0.8vw', cursor='pointer',
                padding='0.8vh 0', letterSpacing='2px', flexGrow=1
            )),
            html.Button('▼  SELL', id='mt5-sell', n_clicks=0, style=dict(
                backgroundColor=RED, border='none', color=WHITE,
                fontFamily='Courier New', fontWeight='bold',
                fontSize='0.8vw', cursor='pointer',
                padding='0.8vh 0', letterSpacing='2px', flexGrow=1
            )),
        ], style=dict(display='flex', gap='0.4vw', marginBottom='0.8vh')),

        # Result
        html.Div(id='mt5-result', style=dict(
            fontSize='0.6vw', letterSpacing='1px',
            padding='0.4vh 0', minHeight='2vh', color=AMBER
        )),

        html.Hr(style=dict(borderColor=DIM, margin='0.5vh 0')),
    ], style=dict(padding='0.8vh 1vw'))


# ── LAYOUT ──────────────────────────────────────────────
app = Dash(suppress_callback_exceptions=True)
app.layout = html.Div([

    dcc.Store(id='watchlist-store',  data=[]),
    dcc.Store(id='chart-type-store', data='line'),
    dcc.Store(id='sidebar-tab',      data='analysis'),
    dcc.Store(id='mt5-connected',    data=False),
    dcc.Interval(id='interval',      interval=1000,  n_intervals=0),
    dcc.Interval(id='mt5-interval',  interval=2000,  n_intervals=0),
    html.Div(id='keyboard-init', style=dict(display='none')),

    # ── HEADER ──────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("XENIA", style=dict(fontSize='1.3vw', fontWeight='bold',
                      color=AMBER, letterSpacing='6px')),
            html.Span("TERMINAL", style=dict(fontSize='0.55vw', color=GREY,
                      letterSpacing='4px', marginLeft='0.6vw',
                      alignSelf='flex-end', paddingBottom='0.1vh'))
        ], style=dict(display='flex', alignItems='baseline')),

        html.Div([
            dcc.Dropdown(options=sp500_tickers, id='stock_drop',
                clearable=False, placeholder='TICKER...',
                style=dict(backgroundColor=BLACK, width='9vw',
                           color=AMBER, fontFamily='Courier New')),
            dcc.Button('SEARCH', id='search-button', n_clicks=0, style=dict(
                backgroundColor=BLACK, width='6vw',
                border=f'1px solid {AMBER}', color=AMBER,
                fontFamily='Courier New', fontSize='0.7vw',
                cursor='pointer', letterSpacing='2px', flexShrink=0)),
            dcc.Loading(type='circle', color=AMBER,
                children=html.Div("AWAITING INPUT", id='signal', style=signal_h_style))
        ], style=dict(display='flex', alignItems='center', gap='0.8vw')),

        html.Div([html.Div(id='time', style=timer_style)])

    ], style=header_style),


    # ── BODY ────────────────────────────────────────────
    html.Div([

        # MAIN PANEL
        html.Div([
            html.Div([
                html.Span("PRICE ACTION"),
                html.Div([
                    html.Button("LINE",   id='btn-line',   n_clicks=0, style=_btn(True)),
                    html.Button("CANDLE", id='btn-candle', n_clicks=0, style=_btn(False)),
                ], style=dict(display='flex', gap='0.4vw'))
            ], style=chart_panel_header_style),
            html.Div(id='main_chart', style=chart_style),
            html.Div([
                html.Div("LIVE NEWS", style=dict(color=GREY, fontSize='0.55vw',
                         letterSpacing='3px', padding='0.4vh 1vw',
                         borderBottom=f'1px solid {DIM}', backgroundColor='#050505')),
                html.Div("NO LIVE NEWS FEED  ──  CONNECT DATA PROVIDER TO ENABLE",
                         style=dict(color=DIM, fontSize='0.65vw',
                                    letterSpacing='2px', padding='1vh 1vw'))
            ], style=news_panel_style),
        ], style=chart_panel_style),

        # SIDEBAR
        html.Div([

            # Tab switcher
            html.Div([
                html.Button("ANALYSIS", id='tab-analysis', n_clicks=0,
                            style=_tab_btn(True)),
                html.Button("TRADE",    id='tab-trade',    n_clicks=0,
                            style=_tab_btn(False)),
            ], style=dict(display='flex', borderBottom=f'1px solid {AMBER}',
                          backgroundColor=DARK, flexShrink=0)),

            # ANALYSIS panel
            html.Div([
                html.Div("SIGNAL PANEL", style=section_label_style),
                html.Div(id='sidebar_content', children=_signal_placeholder(),
                    style=dict(height='35%', overflowY='auto', flexShrink=0,
                               borderBottom=f'1px solid {AMBER}')),
                html.Div("WATCHLIST", style=section_label_style),
                html.Div([
                    html.Div([
                        dcc.Dropdown(options=sp500_tickers, id='watchlist-input',
                            placeholder='ADD TICKER...', clearable=True,
                            style=dict(backgroundColor=BLACK, flexGrow=1,
                                       color=AMBER, fontFamily='Courier New',
                                       fontSize='0.65vw')),
                        html.Button('+', id='watchlist-add-btn', n_clicks=0, style=dict(
                            backgroundColor='transparent', border=f'1px solid {AMBER}',
                            color=AMBER, fontSize='1vw', cursor='pointer',
                            padding='0 0.6vw', fontFamily='Courier New', flexShrink=0))
                    ], style=dict(display='flex', gap='0.4vw',
                                  padding='0.5vh 0.5vw', borderBottom=f'1px solid {DIM}')),
                    html.Div(id='watchlist-display',
                             style=dict(overflowY='auto', flexGrow=1))
                ], style=dict(display='flex', flexDirection='column',
                              flexGrow=1, overflow='hidden'))
            ], id='panel-analysis', style=dict(display='flex', flexDirection='column',
                                               flexGrow=1, overflow='hidden')),

            # TRADE panel
            html.Div([
                # MT5 header + connect
                html.Div([
                    html.Span("METATRADER 5"),
                    html.Button(id='mt5-connect-btn', n_clicks=0,
                                children="CONNECT",
                                style=dict(backgroundColor='transparent',
                                           border=f'1px solid {GREY}', color=GREY,
                                           fontFamily='Courier New', fontSize='0.55vw',
                                           cursor='pointer', padding='0.2vh 0.6vw',
                                           letterSpacing='1px'))
                ], style=section_label_style),

                # Account info
                html.Div(id='mt5-account', children=html.Div(
                    "NOT CONNECTED",
                    style=dict(color=GREY, fontSize='0.6vw',
                               letterSpacing='2px', padding='0.8vh 1vw',
                               borderBottom=f'1px solid {DIM}')
                )),

                # Scrollable trade area
                html.Div([
                    # Order form
                    _order_form(),

                    # Open Positions
                    html.Div([
                        html.Div("OPEN POSITIONS", style=dict(
                            color=AMBER, fontSize='0.55vw', letterSpacing='2px',
                            padding='0.4vh 0.8vw', borderBottom=f'1px solid {DIM}',
                            display='flex', justifyContent='space-between',
                            alignItems='center'
                        )),
                        html.Div(id='mt5-positions'),
                        html.Button("CLOSE ALL POSITIONS",
                            id='mt5-close-all', n_clicks=0,
                            style=dict(backgroundColor='transparent',
                                       border=f'1px solid {RED}', color=RED,
                                       fontFamily='Courier New', fontSize='0.6vw',
                                       cursor='pointer', padding='0.4vh',
                                       width='100%', letterSpacing='2px',
                                       margin='0.5vh 0'))
                    ], style=dict(marginBottom='0.5vh')),

                    # Pending Orders
                    html.Div([
                        html.Div("PENDING ORDERS", style=dict(
                            color=AMBER, fontSize='0.55vw', letterSpacing='2px',
                            padding='0.4vh 0.8vw', borderBottom=f'1px solid {DIM}'
                        )),
                        html.Div(id='mt5-pending')
                    ])

                ], style=dict(overflowY='auto', flexGrow=1))

            ], id='panel-trade', style=dict(display='none', flexDirection='column',
                                             flexGrow=1, overflow='hidden'))

        ], style=sidebar_style)

    ], style=chart_container_style),

    # ── TIMEFRAME BAR ───────────────────────────────────
    html.Div([
        html.Span("TIMEFRAME", style=dict(color=GREY, fontSize='0.65vw',
                  letterSpacing='3px', flexShrink=0)),
        dcc.RadioItems(
            id='time_frame',
            options=['1m', '5m', '15m', '30m', '1h', 'D', 'W', 'M'],
            value='1h',
            style=dict(display='flex', gap='1.5vw'),
            labelStyle=dict(fontFamily='Courier New', color=AMBER,
                            fontSize='0.8vw', letterSpacing='1px'),
            inputStyle=dict(accentColor=AMBER, cursor='pointer')
        )
    ], style=timeframe_bar_style),

    # ── NEWS TICKER ─────────────────────────────────────
    html.Div(
        html.Div(children="", className='news-ticker-content', id='news-line',
            style=dict(color=AMBER, fontSize='0.65vw',
                       fontFamily='Courier New', letterSpacing='1px')),
        style=news_ticker_style
    )

], style=page_style)


# ── CALLBACKS ───────────────────────────────────────────

app.clientside_callback(
    """
    function(n) {
        if (!window._xeniaKey) {
            window._xeniaKey = true;
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    const btn = document.getElementById('search-button');
                    if (btn) btn.click();
                }
            });
        }
        return '';
    }
    """,
    Output('keyboard-init', 'children'),
    Input('interval', 'n_intervals')
)

@callback(Output('time', 'children'), Input('interval', 'n_intervals'))
def update_clock(n):
    return datetime.now().strftime('%H:%M:%S')


# Sidebar tabs
@callback(
    Output('panel-analysis', 'style'),
    Output('panel-trade',    'style'),
    Output('tab-analysis',   'style'),
    Output('tab-trade',      'style'),
    Output('sidebar-tab',    'data'),
    Input('tab-analysis', 'n_clicks'),
    Input('tab-trade',    'n_clicks'),
    prevent_initial_call=True
)
def switch_tab(_, __):
    show = dict(display='flex', flexDirection='column', flexGrow=1, overflow='hidden')
    hide = dict(display='none',  flexDirection='column', flexGrow=1, overflow='hidden')
    if ctx.triggered_id == 'tab-trade':
        return hide, show, _tab_btn(False), _tab_btn(True), 'trade'
    return show, hide, _tab_btn(True), _tab_btn(False), 'analysis'


# Chart type toggle
@callback(
    Output('chart-type-store', 'data'),
    Output('btn-line',   'style'),
    Output('btn-candle', 'style'),
    Input('btn-line',   'n_clicks'),
    Input('btn-candle', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_chart_type(_, __):
    if ctx.triggered_id == 'btn-candle':
        return 'candle', _btn(False), _btn(True)
    return 'line', _btn(True), _btn(False)

# News update
@callback(
    Output('news-line', 'children'), 
    Input('search-button', 'n_clicks'), 
    State('stock_drop', 'value')
)
def update_news(_, value):
    line_news = _news(value)
    return line_news


# Chart update
@callback(
    Output('main_chart', 'children'),
    Input('search-button',    'n_clicks'),
    Input('chart-type-store', 'data'),
    State('stock_drop', 'value'),
    State('time_frame', 'value'),
    prevent_initial_call=False
)
def update_chart(_, chart_type, value, timeframe):
    if not value:
        fig = _base_fig()
        fig.update_layout(title=dict(text='SEARCH TICKER TO DISPLAY CHART',
                          font=dict(size=10, color=GREY), x=0.01))
        return _graph(fig)
    resolution_map = {
        '1m': '1',
        '5m': '5',
        '15m':'15',
        '30m': '30',
        '1h': '60',
        'D': 'D',
        'W': 'W',
        'M': 'M'
    }


    df  = fetcher._fetch_ohlcv_sync(value, resolution=resolution_map.get(timeframe, '1d'))
    fig = _base_fig()
    if chart_type == 'candle' and all(c in df.columns for c in ['open','high','low','close']):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'],
            low=df['low'],   close=df['close'],
            increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
            decreasing=dict(line=dict(color=RED),   fillcolor=RED),
            name=value))
        fig.update_layout(xaxis_rangeslider_visible=False)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['close'].values, mode='lines',
            line=dict(color=AMBER, width=1.2), name=value,
            hovertemplate='%{x}<br>$%{y:.2f}<extra></extra>'))
    fig.update_layout(title=dict(text=f'{value}  ·  {timeframe}',
                      font=dict(size=11, color=AMBER), x=0.01))
    return _graph(fig)


# Signal
@callback(
    Output('signal', 'children'),
    Output('sidebar_content', 'children'),
    Input('search-button', 'n_clicks'),
    State('stock_drop', 'value'),
    prevent_initial_call=True
)
async def get_signal(_, values):
    if not values:
        return "AWAITING INPUT", _signal_placeholder()
    system = create_default_system([values])
    await system.fetch_all_data()
    await system.train_all_models()
    current_idx = 0
    for i, _ in enumerate(system.data[values].index):
        current_idx = i
    combined_signal, combined_confidence = system.get_combined_signal(values, current_idx)
    recmd  = system.get_recommendation(combined_signal, combined_confidence)
    sig    = recmd.upper()
    color  = GREEN if 'BUY' in sig else RED if 'SELL' in sig else AMBER
    header = html.Span(sig, style=dict(color=color, letterSpacing='2px'))
    return header, _signal_panel(values, combined_signal, combined_confidence, recmd)


# Auto-fill MT5 symbol from chart ticker
@callback(
    Output('mt5-symbol', 'value'),
    Input('stock_drop', 'value'),
    prevent_initial_call=True
)
def sync_symbol(value):
    return value or dash.no_update


# Show/hide price input for limit/stop orders
@callback(
    Output('mt5-price-row', 'style'),
    Input('mt5-order-type', 'value')
)
def toggle_price_row(order_type):
    if order_type in ('limit', 'stop'):
        return dict(marginBottom='0.6vh', display='block')
    return dict(marginBottom='0.6vh', display='none')


# MT5 connect / disconnect
@callback(
    Output('mt5-connected',    'data'),
    Output('mt5-connect-btn',  'children'),
    Output('mt5-connect-btn',  'style'),
    Input('mt5-connect-btn',   'n_clicks'),
    State('mt5-connected',     'data'),
    prevent_initial_call=True
)
def toggle_mt5(_, connected):
    if connected:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        return False, "CONNECT", dict(
            backgroundColor='transparent', border=f'1px solid {GREY}', color=GREY,
            fontFamily='Courier New', fontSize='0.55vw', cursor='pointer',
            padding='0.2vh 0.6vw', letterSpacing='1px')
    ok, _ = _mt5_init()
    if ok:
        return True, "DISCONNECT", dict(
            backgroundColor='transparent', border=f'1px solid {GREEN}', color=GREEN,
            fontFamily='Courier New', fontSize='0.55vw', cursor='pointer',
            padding='0.2vh 0.6vw', letterSpacing='1px')
    return False, "FAILED", dict(
        backgroundColor='transparent', border=f'1px solid {RED}', color=RED,
        fontFamily='Courier New', fontSize='0.55vw', cursor='pointer',
        padding='0.2vh 0.6vw', letterSpacing='1px')


# Refresh account + positions + pending on interval
@callback(
    Output('mt5-account',  'children'),
    Output('mt5-positions', 'children'),
    Output('mt5-pending',   'children'),
    Input('mt5-interval',  'n_intervals'),
    State('mt5-connected', 'data')
)
def refresh_mt5(_, connected):
    if not connected:
        offline = html.Div("NOT CONNECTED",
                           style=dict(color=GREY, fontSize='0.6vw',
                                      letterSpacing='2px', padding='0.8vh 1vw',
                                      borderBottom=f'1px solid {DIM}'))
        return offline, _render_positions([]), _render_pending([])
    return (_render_account(_mt5_account()),
            _render_positions(_mt5_positions()),
            _render_pending(_mt5_orders()))


# BUY order
@callback(
    Output('mt5-result', 'children', allow_duplicate=True),
    Input('mt5-buy', 'n_clicks'),
    State('mt5-symbol',     'value'),
    State('mt5-lots',       'value'),
    State('mt5-order-type', 'value'),
    State('mt5-price',      'value'),
    State('mt5-sl',         'value'),
    State('mt5-tp',         'value'),
    State('mt5-connected',  'data'),
    prevent_initial_call=True
)
def place_buy(_, symbol, lots, order_type, price, sl, tp, connected):
    if not connected:
        return html.Span("NOT CONNECTED", style=dict(color=RED))
    if not symbol or not lots:
        return html.Span("FILL SYMBOL + LOTS", style=dict(color=RED))
    ok, msg = _mt5_send(symbol, 'buy', lots, sl or 0, tp or 0, order_type, price or 0)
    return html.Span(msg, style=dict(color=GREEN if ok else RED))


# SELL order
@callback(
    Output('mt5-result', 'children', allow_duplicate=True),
    Input('mt5-sell', 'n_clicks'),
    State('mt5-symbol',     'value'),
    State('mt5-lots',       'value'),
    State('mt5-order-type', 'value'),
    State('mt5-price',      'value'),
    State('mt5-sl',         'value'),
    State('mt5-tp',         'value'),
    State('mt5-connected',  'data'),
    prevent_initial_call=True
)
def place_sell(_, symbol, lots, order_type, price, sl, tp, connected):
    if not connected:
        return html.Span("NOT CONNECTED", style=dict(color=RED))
    if not symbol or not lots:
        return html.Span("FILL SYMBOL + LOTS", style=dict(color=RED))
    ok, msg = _mt5_send(symbol, 'sell', lots, sl or 0, tp or 0, order_type, price or 0)
    return html.Span(msg, style=dict(color=GREEN if ok else RED))


# Close single position
@callback(
    Output('mt5-result', 'children', allow_duplicate=True),
    Input({'type': 'close-pos', 'index': ALL}, 'n_clicks'),
    State('mt5-connected', 'data'),
    prevent_initial_call=True
)
def close_position(n_clicks_list, connected):
    if not ctx.triggered_id or not any(n_clicks_list):
        return dash.no_update
    if not connected:
        return html.Span("NOT CONNECTED", style=dict(color=RED))
    ok, msg = _mt5_close(ctx.triggered_id['index'])
    return html.Span(msg, style=dict(color=GREEN if ok else RED))


# Close all positions
@callback(
    Output('mt5-result', 'children', allow_duplicate=True),
    Input('mt5-close-all', 'n_clicks'),
    State('mt5-connected',  'data'),
    prevent_initial_call=True
)
def close_all(_, connected):
    if not connected:
        return html.Span("NOT CONNECTED", style=dict(color=RED))
    results = _mt5_close_all()
    if not results:
        return html.Span("NO POSITIONS TO CLOSE", style=dict(color=GREY))
    return html.Span(' | '.join(results), style=dict(color=GREEN))


# Cancel pending order
@callback(
    Output('mt5-result', 'children', allow_duplicate=True),
    Input({'type': 'cancel-order', 'index': ALL}, 'n_clicks'),
    State('mt5-connected', 'data'),
    prevent_initial_call=True
)
def cancel_order(n_clicks_list, connected):
    if not ctx.triggered_id or not any(n_clicks_list):
        return dash.no_update
    if not connected:
        return html.Span("NOT CONNECTED", style=dict(color=RED))
    ok, msg = _mt5_cancel_order(ctx.triggered_id['index'])
    return html.Span(msg, style=dict(color=GREEN if ok else RED))


# Watchlist
@callback(
    Output('watchlist-store', 'data'),
    Input('watchlist-add-btn', 'n_clicks'),
    State('watchlist-input',  'value'),
    State('watchlist-store',  'data'),
    prevent_initial_call=True
)
def add_to_watchlist(_, ticker, watchlist):
    if ticker and ticker.upper() not in [t.upper() for t in watchlist]:
        return watchlist + [ticker.upper()]
    return watchlist

@callback(
    Output('watchlist-store', 'data', allow_duplicate=True),
    Input({'type': 'remove-btn', 'index': ALL}, 'n_clicks'),
    State('watchlist-store', 'data'),
    prevent_initial_call=True
)
def remove_from_watchlist(n_clicks_list, watchlist):
    if not ctx.triggered_id or not any(n_clicks_list):
        return watchlist
    return [t for t in watchlist if t != ctx.triggered_id['index']]

@callback(
    Output('stock_drop', 'value'),
    Input({'type': 'watchlist-item', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def load_from_watchlist(n_clicks_list):
    if not ctx.triggered_id or not any(n_clicks_list):
        return dash.no_update
    return ctx.triggered_id['index']

@callback(
    Output('watchlist-display', 'children'),
    Input('watchlist-store', 'data')
)
def render_watchlist(watchlist):
    if not watchlist:
        return html.Div("ADD TICKERS TO WATCH",
            style=dict(color=GREY, fontSize='0.6vw',
                       letterSpacing='2px', padding='1vh 1vw'))
    return [_watchlist_row(t) for t in watchlist]


if __name__ == '__main__':
    app.run(debug=False)   # debug=False — MT5 is not thread-safe