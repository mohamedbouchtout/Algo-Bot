"""Quick debug script to test IB historical data fetching"""
import sys
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

print(f"Python: {sys.version}")

from ib_insync import IB, Stock, util
print("ib_insync loaded OK")

ib = IB()
print(f"RequestTimeout: {IB.RequestTimeout}")

try:
    print("Attempting connect on port 4002...")
    ib.connect('127.0.0.1', 4002, clientId=99)
    print("Connected!")

    ib.reqMarketDataType(3)
    print("Set market data type to 3 (delayed)")

    contract = Stock('AAPL', 'SMART', 'USD')
    qualified = ib.qualifyContracts(contract)
    print(f"qualifyContracts returned: {qualified}, conId={contract.conId}")

    print("Requesting historical data for AAPL (250 D, 1 day bars)...")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='250 D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    print(f"Got {len(bars)} bars for AAPL")

    if bars:
        df = util.df(bars)
        print(f"DataFrame shape: {df.shape}")
        print(df.tail(3))
    else:
        print("WARNING: No bars returned!")

    ib.disconnect()
    print("Done - disconnected")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    try:
        ib.disconnect()
    except:
        pass
