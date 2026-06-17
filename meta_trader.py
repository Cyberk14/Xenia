
import MetaTrader5 as mt5

mt5.initialize()

print(mt5.terminal_info())
print(mt5.account_info())

print(mt5.symbols_total())
symbol_turple = mt5.symbols_get()

symbol_names = [{"NAME": asset.name,
                 "DECRIPTION": asset.description,
                 'FILLING': mt5.symbol_info(asset.name).filling_mode} for asset in symbol_turple]

print(symbol_names)

mt5.last_error()