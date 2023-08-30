# Deformable Convolution Care Around
## Abstract
The Def_conv v3 was published with "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions" in CVPR2023.

Things thay did inspired me to do this project. Def_conv can get long-range dependence, and have adaptive spatial aggregation. All of that they did was so great, but I rethinking about some quations.

![image](https://github.com/okok009/def_conv_CareAround/imgs/Figure2.def_conv.jpg)

The reason of convolution works is that always find some relations between the core and around pixels, but if we use Def_conv, we might lost this ralation around the core. So I think maybe we can use Original Convolution and Deformable Convolution collateral, some good things might happen.

![image](https://github.com/okok009/def_conv_CareAround/imgs/Figure2.mine.jpg)

Unluckly, things not doing well.

Mine thought are premature, residual network might already skip this issue. And the thing I did exactly need more memory.

But in the end, I still want to upload this project. Have fun!
## 使用方法
雖然結果不好，但還是可以使用看看。

需要先將Dataset 放置好，並將Annotation 的資訊轉換為檔案中train2017.txt 中的形式，以利dataloader 讀取圖片與資訊。

coco_annotation.py 是專門為了coco2017 中的annotation.json 檔所設計的，雖然需要跑很多時間但還算堪用。如果是使用其他檔案，必須自己轉換為txt 檔，或是用其他dataloader 來讀取檔案。

將檔案都放置完後，只須執行train 即可進行訓練。測試的程式都還沒寫好，可望之後有時間再來完成，但目前覺得這個方向沒什麼機會成功，暫且先擱置在這。
