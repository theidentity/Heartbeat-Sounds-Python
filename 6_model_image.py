from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D,Input
from keras.models import Model,save_model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50,Xception,InceptionResNetV2

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
# from matplotlib import pyplot as plt


class ImageClassifier(object):
	"""docstring for ImageClassifier"""
	def __init__(self):

		# self.img_rows,self.img_cols = 197,197//ResNet50,Xception,
		self.img_rows,self.img_cols = 512,512
		self.batch_size = 4
		self.seed = 42
		self.input_shape = (self.img_rows,self.img_cols,3)
		self.num_classes = 4
		self.steps_per_epoch = 1465//self.batch_size,
		self.validation_steps = 1468//self.batch_size,

		self.train_img_path = 'data/image_folds/fold1/train/'
		self.validation_img_path = 'data/image_folds/fold1/validation/'

		self.name = 'ResNet50_fold1'
		self.save_path = ''.join(['models/',self.name,'_best','.h5'])

		self.model = self.get_model()
		print self.model.summary()
		

	def get_model(self):

		base_model = ResNet50(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = Xception(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = InceptionResNetV2(include_top=False,input_shape=self.input_shape,weights='imagenet')

		x = base_model.output
		x = Flatten()(x)
		x = Dense(1024,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(1024,activation='relu')(x)
		x = Dropout(0.5)(x)
		predictions = Dense(self.num_classes,activation='softmax')(x)
		
		model = Model(base_model.inputs,predictions)
		return model

	def build_model(self,lr=1e-4):
		
		opt = Adam(lr=lr)
		self.model.compile(
			optimizer = opt,
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)

	def get_train_generator(self,path):
		img_gen = ImageDataGenerator(
			zoom_range = 0.0,
			width_shift_range = 0.2,
			height_shift_range = 0.0,
			horizontal_flip = False,
			rotation_range = 0.0,
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb'
			)

		return img_gen

	def get_validation_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = False
			)

		return img_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		
		# tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		# return [early_stopping,checkpointer,tensorboard]
		return [early_stopping,checkpointer]

	def train(self,lr=1e-4,num_epochs=2):

		self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def continue_training(self,lr=1e-4,num_epochs=10):

		self.model = load_model(self.save_path)
		self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def normalize_array(self,arr,lower=0,upper=255,mode='minmax'):
		arr = np.array(arr,dtype=np.float32)
		if mode == 'minmax':
			arr = (upper-lower)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))
		else:
			arr = (arr-np.mean(arr))/np.std(arr)
		return arr

	def get_predictions(self,save=True):
		
		self.model = load_model(self.save_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)
		# validation_generator = self.get_train_generator(self.train_img_path)

		validation_steps = self.validation_steps[0]
		
		y_actual = np.empty((0,self.num_classes))
		y_pred = np.empty((0,self.num_classes))

		for i in range(validation_steps+1):
			
			X,y = validation_generator.next()
			y_actual = np.vstack([y_actual,y])
			
			pred = self.model.predict(X,
				batch_size = self.batch_size,
				verbose = 1)
			y_pred = np.vstack([y_pred,pred])

		if save:
			np.save('tmp/'+self.name+'_pred.npy',y_pred)
			np.save('tmp/'+self.name+'_true.npy',y_actual)

	def get_metrics(self):
		y_pred = np.load('tmp/'+self.name+'_pred.npy')
		y_actual = np.load('tmp/'+self.name+'_true.npy')
		
		y_actual = np.argmax(y_actual,axis=1)
		y_pred = np.argmax(y_pred,axis=1)


		cm = confusion_matrix(y_actual,y_pred)
		report = classification_report(y_actual,y_pred)
		accuracy = accuracy_score(y_actual,y_pred)

		print cm
		print report
		print 'Accuracy :',accuracy


if __name__ == '__main__':
	m1 = ImageClassifier()
	m1.train(lr=1e-4,num_epochs=20)
	# m1.continue_training(lr=1e-4,num_epochs=20)
	m1.get_predictions(save=True)
	m1.get_metrics()

