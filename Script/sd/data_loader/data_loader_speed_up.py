from tqdm import tqdm
import aiohttp
import asyncio
import os
import pandas as pd
from aiofiles import open as aio_open

base_folder_location = "E:/"
all_image_path = os.path.join(base_folder_location, "image")
excel_path = os.path.join(base_folder_location, "excel")

# Ingredient layer 1 information JSON to DataFrame conversion
excel_1_path = os.path.join(excel_path, "df_with_image_id_and_url.xlsx")
df_with_image_id_and_url = pd.read_excel(excel_1_path, engine="openpyxl")
all_image_information = [row.to_dict() for _, row in df_with_image_id_and_url.iterrows()]


def chunked(iterable, size):
    """Split an iterable into chunks of size."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


async def download_image(session, image_url_data, image_path):
    image_url_info = str(image_url_data['image_url'])
    image_id_info = str(image_url_data['image_id'])
    new_filename = image_id_info
    try:
        image_download_path = os.path.join(image_path, new_filename)
        async with session.get(image_url_info, timeout=300) as response:
            if response.status == 200:
                async with aio_open(image_download_path, mode='wb') as f:
                    async for data in response.content.iter_any():
                        await f.write(data)
                return {'image_id': new_filename, 'image_download_status': "Download Success"}
            else:
                return {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': response.status}
    except aiohttp.ClientError as e:
        return {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': str(e)}
    except Exception as e:
        return {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': str(e)}


async def download_images(all_image_url_data_list, image_path):
    concurrency = 100  # Increase the concurrency level to 100
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, image_url_data, image_path) for image_url_data in all_image_url_data_list]

        with tqdm(total=len(all_image_url_data_list)) as pbar:
            results = []
            for coro_batch in chunked(tasks, concurrency):
                coro_results = await asyncio.gather(*coro_batch)
                results.extend(coro_results)
                pbar.update(len(coro_results))  # Update progress bar
            return results


async def main():
    await download_images(all_image_information, all_image_path)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
