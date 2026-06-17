from XeniaV2 import create_default_system
import sys
import asyncio

async def get_signal(values):
    system = create_default_system(values)
    await system.fetch_all_data()
    await system.train_all_models()

    for value in values:
        for i, date in enumerate(system.data[value].index):
            current_idx = i

        combined_signal, combined_confidence = system.get_combined_signal(value, current_idx)
        print(combined_signal, ': ', combined_confidence)
        recmd = system.get_recommendation(combined_signal, combined_confidence)
        print(recmd)




if __name__ == "__main__":
    asyncio.run(get_signal(['TSLA']))

