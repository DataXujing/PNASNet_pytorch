import cv2
import os


def checkimg(src,mode="train"):

    for files in os.walk(src):
        if len(files[2]) == 0:
            continue
        else:
            base_path = files[0]
            # save_path = base_path.replace("data","hist")
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            for file in files[2]:
                new_name = "".join(file.split(".")[:-1])+".jpg"
                
                img_path = os.path.join(base_path,file)

                img = cv2.imread(img_path)

                if not os.path.exists("./data/"+mode):
                    os.makedirs("./data/"+mode)

                cv2.imwrite("./data/"+mode+"/"+new_name,img)
                print(new_name)


if __name__ == "__main__":
    # checkimg("")
    pass