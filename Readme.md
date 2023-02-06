############################################## README FILE ##############################################



- Authors are Trinh Duc Tinh(@groutlloyd), Duong Vo, Van Nhan, Truong Giang in HCMUT. August, 2021.



- How to use and configure the project:

Step 1: Using pip command to setup all packages needed in requirement-package.txt.

Step 2: Open configuration.txt and change the "data_dir" path into your dataset path. Inside it should 
have the folder that keeps all the folder have respectively type of woods.

Also you can change many things in there like CNN types, amount of classes and lots of other parameters.

E.g. D:/Dataset have 13 classes(folders), which are Pine, Maple, Oak,....  

Step 3: Run /Prepare Data And Label/GetFolds.py to seperate folder into numbers of different folders.
That helps generating Data easier and more flexible.

Step 4: Run /Prepare Data And Label/GenerateData.py .This file make train and test set.

Step 5: Run /Train and Tests/TrainNetwork.py .The name say it all. Can config some other things like,
validation/train ratio, augmentation setup,...

Step 6: Finally! Run /Train and Tests/TestPatch.py to get the Patch accuracy test and TestImage.py to 
check Top-1 Accuracy and Top-3 Accuracy. Huge amounts of infos appear in the running console, too!
Steadily check all of those things to have further results and benchmarks of all the CNNs.


- That's probably the last project that these guys above can have the chance to cooperate. A little sad. 

But, we are youngs and life just start for us. No one can know what crazy things will happen with 
our life.

Thanks to our advisor - Teacher Pham Viet Cuong for letting us the chance to make this project.

I and all the members wish you - person that open this repo - the best things in life.
