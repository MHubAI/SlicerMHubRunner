![MHub Runner (3DSlicer Plugin)](https://github.com/AIM-Harvard/SlicerMHubRunner/blob/main/MRunner/Resources/Icons/Name.png?raw=true)

The MHubRunner extension seamlessly integrates Deep Learning models from [MHub.ai](https://mhub.ai) into 3D Slicer.

## MHub.ai
MHub is a repository for Machine Learning models for medical imaging.
The goal of mhub is to make these models universally accessible by containerizing the entire model pipeline and standardizing the I/O interface.

Find out more at [mhub.ai](https://mhub.ai) and check out the mhub [GitHub repository](https://github.com/MHubAI).

## Requirements
We only need [Docker](https://docs.docker.com/get-docker/) 🐳 to be installed on your system. That's it.

# Usage

First, open a volume in Slicer on which you want to run the plugin.
You can use your own data or the slicer sample data.
Slicer sample data can be found at *File > Download Sample Data*, to download a chest CT scan click on *CTChest*.
Now open the *MRunner* module (navigate to *3D Slicer > Modules > Examples > MHubRunner*).
You will now see the graphical user interface (GUI) of the module.

<img width="602" alt="Bildschirmfoto 2025-01-27 um 11 11 28" src="https://github.com/user-attachments/assets/2d8ba82e-a6f2-41c9-8c57-12cc3418bc77" />

## MHub Model

Here you see an overview of all available MHub.ai models.
You can use the search box to find models suitable for your use case.

Additional model information such as a short description of the model, the modalities and the expected input data can be displayed using the details button.
For more information, you can open the model card on MHub.ai using the web button which will take you directly to the model page.

## Input Image

Here you can select and inspect the input image to run the model on.

## Backend

Under the backend tab, you can select the backend (Docker by default) and manage your local model images.
You can select an image and use the update button to pull the latest version of this model.
You can use the delete button to remove the image to free up space on your disk.
You can always download the image again.

We recommend running all MHub.ai models using the Docker backend. For Linux users, we are currently exploring udocker as an alternative backend.


## GPU

Some of our models require a GPU to run, most will be significantly faster with a GPU available. If you have a supported GPU installed, it will be listed here.
You can then select the gpu(s) you want to use running the model here.

## Advanced Options

You can manually specify the path to the Docker executable, kill all running background processes and see the run log under the advanced options.

# Important Note

**This repository and plugin are under active development, as is the mhub repository.
Use this plugin with caution and always backup your data. We strongly recommend that you only use the slicer sample data and only use this plugin in a non-production environment.**
