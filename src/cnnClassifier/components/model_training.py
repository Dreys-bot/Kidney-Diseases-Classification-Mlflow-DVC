
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    #Load pretrained model VGG16
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):

        #Define datagenerator arguments for validation
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        #Define params for create batch data of train or validation dataset
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1], #(224, 224)
            batch_size=self.config.params_batch_size, #32
            interpolation="bilinear"
        )

        #Create 20% of training data for validation dataset by rescaling 
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        #Create many batch of validation dataset
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data, #Take training_data
            subset="validation", #Take Validation dataset
            shuffle=False,
            **dataflow_kwargs #create batch of 32 samples with size (224, 224)
        )

        # Create train datagenerator also
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs #rescale and take 80% of all data
            )
        else:
            train_datagenerator = valid_datagenerator

        #Processing and create batch of 32 
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

        
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )