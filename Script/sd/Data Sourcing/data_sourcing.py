import json
import pandas as pd
import re
import aiohttp
import asyncio
import aiofiles
from tqdm import tqdm
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
base_dataset_folder_location = "D:/DDPM/Food-Vision/dataset"
image_location = os.path.join(base_dataset_folder_location, "image")


def check_each_image_download_status(data):
    image_global_path = os.path.join(image_location, data)
    if os.path.isfile(image_global_path):
        image_content = cv2.imread(image_global_path)
        if str(type(image_content)) == "<class 'NoneType'>":
            return "Download Error"
        else:
            return "Download Success"
    else:
        return "Download Failed"


async def fetch_data_from_multiple_url(urls):
    local_result = []
    semaphore = asyncio.Semaphore(500)
    tasks = [get_image_from_url(url.to_dict(), semaphore) for url in tqdm(urls)]
    try:
        result_tuples = [await f for f in tqdm(asyncio.as_completed(tasks), total=(len(tasks)))]
        for result_tuple in result_tuples:
            local_result.append(result_tuple)
    except Exception as exception:
        exception = str(exception)
    return local_result


async def get_image_from_url(image_url_data, semaphore):
    image_url_info = str(image_url_data['image_url'])
    image_id_info = str(image_url_data['image_id'])
    new_filename = image_id_info
    return_dic = {'image_id': new_filename, 'image_download_status': "Download Failed", 'request_response_status': ""}
    download_dir = image_location
    image_download_path = os.path.join(download_dir, new_filename)
    content_not_found_flag = False
    retries = 0
    while retries < 3:
        try:
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url_info) as response:
                        try:
                            if response.status == 200:
                                f = await aiofiles.open(image_download_path, mode='wb')
                                await f.write(await response.read())
                                await f.close()
                                content_not_found_flag = True

                                if content_not_found_flag:
                                    return_dic["image_download_status"] = "Download Failed"
                                    return_dic["request_response_status"] = str(response.status)
                                else:
                                    return_dic["image_download_status"] = "Download Success"
                                    return_dic["request_response_status"] = str(response.status)
                            else:
                                return_dic["image_download_status"] = "Download Failed"
                                return_dic["request_response_status"] = str(response.status)
                        except Exception as timeout_exception:
                            timeout_exception = str(timeout_exception)
                            return_dic["image_download_status"] = "Download Failed"
                            return_dic["request_response_status"] = str(response.status)

        except aiohttp.ClientConnectorError:
            pass
        except ConnectionResetError:
            pass
        retries += 1
        await asyncio.sleep(1)
    return [return_dic]


def check_image_status(list_of_dict):
    if len(list_of_dict) != 0:
        image_info_dict = {'id': list_of_dict[0]['id'], 'url': list_of_dict[0]['url']}
        return image_info_dict
    else:
        return "Empty"


def get_image_url(image_info_dict):
    if len(image_info_dict) != 0:
        if 'url' in image_info_dict:
            return image_info_dict['url']
        else:
            return "Empty"
    else:
        return "Empty"


def get_image_id(image_info_dict):
    if len(image_info_dict) != 0:
        if 'id' in image_info_dict:
            return image_info_dict['id']
        else:
            return "Empty"
    else:
        return "Empty"


def chunk_preprocessing(layer_1_json_path: str, layer_2_json_path: str, chunk_size: int):
    with open(layer_2_json_path, 'r') as file:
        layer_2_json = json.load(file)

    with open(layer_1_json_path, 'r') as file:
        layer_1_json = json.load(file)

    # Ingredient layer 1 information JSON to Dataframe conversion
    ingredient_layer_1_information_df = pd.DataFrame(layer_1_json)

    # Ingredient layer 2 information JSON to Dataframe conversion
    ingredient_layer_2_information_df = pd.DataFrame(layer_2_json)
    del layer_1_json, layer_2_json
    master_ingredient_df = pd.merge(ingredient_layer_1_information_df, ingredient_layer_2_information_df, on=['id'])
    master_ingredient_df = master_ingredient_df[['id', 'ingredients', 'title', 'instructions', 'partition', 'images']]
    master_ingredient_df_train = master_ingredient_df[master_ingredient_df['partition'] == "train"]

    master_ingredient_df_train['Image_Status'] = master_ingredient_df_train['images'].apply(check_image_status)

    master_ingredient_df_train_empty_image_info = master_ingredient_df_train[master_ingredient_df_train['Image_Status'] == "Empty"]
    print(master_ingredient_df_train_empty_image_info.shape)

    master_ingredient_df_train['image_id'] = master_ingredient_df_train['Image_Status'].apply(get_image_id)
    master_ingredient_df_train['image_url'] = master_ingredient_df_train['Image_Status'].apply(get_image_url)

    master_ingredient_df_train_empty_image_info_id = master_ingredient_df_train[master_ingredient_df_train['image_id'] == "Empty"]

    master_ingredient_df_train_first_slot = master_ingredient_df_train.head(chunk_size)
    master_ingredient_df_train_first_slot = master_ingredient_df_train_first_slot.reset_index().drop(['index'], axis=1)
    all_image_information = (row for _, row in master_ingredient_df_train_first_slot.iterrows())

    loop = asyncio.get_event_loop()
    modify_results = loop.run_until_complete(fetch_data_from_multiple_url(all_image_information))

    master_ingredient_df_train_first_slot = master_ingredient_df_train.head(50000)
    master_ingredient_df_train_first_slot = master_ingredient_df_train_first_slot.reset_index().drop(['index'], axis=1)

    master_ingredient_df_train_first_slot['image_download_status'] = master_ingredient_df_train_first_slot['image_id'].swifter.apply(check_each_image_download_status)

    master_ingredient_df_train_1_error = master_ingredient_df_train_first_slot[master_ingredient_df_train_first_slot['image_download_status'] == "Download Error"]
    master_ingredient_df_train_1_failed = master_ingredient_df_train_first_slot[master_ingredient_df_train_first_slot['image_download_status'] == "Download Failed"]
    master_ingredient_df_train_1_success = master_ingredient_df_train_first_slot[master_ingredient_df_train_first_slot['image_download_status'] == "Download Success"]

    print(master_ingredient_df_train_1_error.shape)
    print(master_ingredient_df_train_1_failed.shape)
    print(master_ingredient_df_train_1_success.shape)

    # Make Final Image
    expired_image_location = list(master_ingredient_df_train_1_error['image_id'])
    try:
        for expired_image in expired_image_location:
            os.remove(os.path.join(image_location, expired_image))
    except FileNotFoundError:
        pass

    # Saving First 50000 Image, Ingredient combination in Excel For Further Process
    excel_1_path = os.path.join(base_dataset_folder_location, "excel/first_image_ingredient_pair.xlsx")
    master_ingredient_df_train_1_success.to_excel(excel_1_path, engine='openpyxl', index=False)


if __name__ == '__main__':
    layer_1_path = 'D:/DDPM/Food-Vision/dataset/layer1.json'
    layer_2_path = 'D:/DDPM/Food-Vision/dataset/layer2.json'
    chunk_amount = 50000
    chunk_preprocessing(layer_1_path, layer_2_path, chunk_amount)


