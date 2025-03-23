import os

def delete_all_files_in_folder():
    folderPath1 = "E:/Users/sj879/Desktop/vscode/Research_in_Physics/cat_picture"
    folderPath2 = "E:/Users/sj879/Desktop/vscode/Research_in_Physics/dog_picture"
    
    folder_paths = [folderPath1, folderPath2]
    # 지정된 폴더 내의 모든 파일 삭제
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # 파일인지 확인 후 삭제
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")


if __name__ == "__main__":
    delete_all_files_in_folder()