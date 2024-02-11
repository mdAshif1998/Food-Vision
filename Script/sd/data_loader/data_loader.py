from tqdm import tqdm
import aiohttp
import asyncio
import warnings
import os
import pandas as pd

# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
base_folder_location = "E:/"
all_image_path = os.path.join(base_folder_location, "image")
excel_path = os.path.join(base_folder_location, "excel")

# Ingredient layer 1 information JSON to DataFrame conversion
excel_1_path = os.path.join(excel_path, "df_with_image_id_and_url.xlsx")
df_with_image_id_and_url = pd.read_excel(excel_1_path, engine="openpyxl")
all_image_information = [row.to_dict() for _, row in df_with_image_id_and_url.iterrows()]


async def download_image(session, image_url_data, image_path):
    image_url_info = str(image_url_data['image_url'])
    image_id_info = str(image_url_data['image_id'])
    new_filename = image_id_info
    try:
        image_download_path = os.path.join(image_path, new_filename)
        async with session.get(image_url_info, timeout=300) as response:
            if response.status == 200:
                with open(image_download_path, "wb") as f:
                    async for data in response.content.iter_chunked(1024):
                        f.write(data)
                return {'image_id': new_filename, 'image_download_status': "Download Success"}
            else:
                return {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': response.status}
    except Exception as e:
        return {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': str(e)}


async def download_images(all_image_url_data_list, image_path):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for image_url_data in all_image_url_data_list:
            tasks.append(download_image(session, image_url_data, image_path))

        with tqdm(total=len(all_image_url_data_list)) as pbar:
            results = []
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update()  # Update progress bar for each completed task
            return results


# Example usage:
async def main():
    await download_images(all_image_information, all_image_path)


if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        loop = asyncio.get_event_loop()
        loop.create_task(main())
    else:
        asyncio.get_event_loop().run_until_complete(main())
