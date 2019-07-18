import pandas


first_cattle_file_path = 'D:\\bengal_meat\\cattle_2.csv'
first_photo_file_path = 'D:\\bengal_meat\\photo_gallery_2.csv'
first_db_output_path = 'D:\\bengal_meat\\first_db.csv'


second_cattle_file_path = 'D:\\bengal_meat\\cattle.csv'
second_photo_file_path = 'D:\\bengal_meat\\photo_gallery.csv'
second_db_output_path = 'D:\\bengal_meat\\second_db.csv'


def collect_cattle_photos_name(cattle_file, photo_file):
    cattle = pandas.read_csv(cattle_file)
    cattle_photos = pandas.read_csv(photo_file)
    cattle_with_photo = pandas.merge(cattle, cattle_photos, on='cattle_id')
    cattle_with_photo = cattle_with_photo[['cattle_id', 'sku', 'weight', 'breeds', 'picture']]
    cattle_with_photo = cattle_with_photo.dropna()
    return cattle_with_photo


def dump_into_csv(cattle_file, photo_file, output_file_path):
    cattle_with_photo = collect_cattle_photos_name(cattle_file, photo_file)
    cattle_with_photo.to_csv(output_file_path, index=False)


if __name__ == '__main__':
    dump_into_csv(first_cattle_file_path, first_photo_file_path, first_db_output_path)
    dump_into_csv(second_cattle_file_path, second_photo_file_path, second_db_output_path)
