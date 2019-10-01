"""Implementation of CNN style transfer. Generate an image with the same "content" as img1 with the
"style" of img2. The network operates by optimizing with respect to the input img2. The network has
 already been optimized with respect to the input parameters
 reproducing content: a network has been optimized to capture the features of an input image
 reproducing style:
 Gram Matrix: 1/N*X.X^T - this is the "autocorrelation". Used to calculate the MSE as a measure of
 style loss. It's dimension are featuresXfeatures - without the "spatial dimension" dimensions
 The last step is to put the two (weighted) losses together
 Three inputs: content image, style image and optimization is done with respect to a weighted combination
 of content loss and style loss

"""
import numpy as np
import datetime

# GUI imports
import tkinter as tk
from tkinter import filedialog
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# CNN imports
#from keras.layers import Input, Lambda, Dense, Flatten
#keras
from keras.layers import AveragePooling2D, MaxPool2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
# optimization
from scipy.optimize import fmin_l_bfgs_b


class StyleTransfer:
    def __init__(self):
        # Supported Transfer models
        # TODO test with additional models
        self.models = {'VGG16'}
        self.processes = {'transfer', 'content', 'style'}
        # initialize blank source content and style images
        self.content_image = np.zeros(shape=(100, 100, 3), dtype='uint8')
        self.style_image = np.zeros(shape=(100, 100, 3), dtype='uint8')
        self.image_size = [None, None]  # width, height
        # initialize gui
        # after model is selected find the number of conv layer and set the max value for cutoff
        self.default_iterations = 10
        self.default_cutoff = 11  # there are 13 convolutions in total any of which can be selected as output for "content"
        self.debug = True  # debug flag
        # start GUI
        self.initialize_gui()

    def gui_transfer(self):
        """Transfer style of style_image to the content of the content_image (GUI command).
        ONLY VGG16 !!
        """

        self.update_status_text()
        self.update_status_text('Starting Transfer')
        # pre process content and style images
        # convert to array
        content_image = image.img_to_array(self.content_image)
        # add extra dimension for for a "1-dim" batch (tf backend)
        content_image = np.expand_dims(content_image, axis=0)
        # pre process for VGG (theano?)
        content_image = preprocess_input(content_image)
        # same for style images
        style_image = image.img_to_array(self.style_image)
        style_image = np.expand_dims(style_image, axis=0)
        style_image = preprocess_input(style_image)
        batch_shape = style_image.shape
        if self.debug:
            print('finished image preprocessing')
        # create VGG16 CNN to extract features across the entire image
        input_shape = [self.image_size[1], self.image_size[0], 3]
        vgg = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
        if self.debug:
            print('loaded vgg')
        # recreate VGG by replacing maxpooling layers with average pooling layers
        # to keep all information in the image (tested by original author)
        vgg_averagepool = Sequential()
        # input = vgg.layers[0].input
        for layer in vgg.layers:
            if layer.__class__ == MaxPool2D:
                vgg_averagepool.add(AveragePooling2D())
            else:
                vgg_averagepool.add(layer)
        if self.debug:
            print('created vgg_averagepool')
        # get GUI content cutoff value
        content_cutoff_val = int(self.cutoff_val.get())
        # create content model
        content_model = Model(inputs=vgg_averagepool.input, outputs=vgg_averagepool.layers[content_cutoff_val].get_output_at(0))
        if self.debug:
            content_model.summary()
        self.content_target = K.variable(content_model.predict(content_image))
        # create the style model
        # target outputs the first convolution at each block of convolutions
        # select output at index 1, since outputs at index 0 correspond to the original
        # vgg with maxpool
        self.symbolic_conv_outputs = [
            layer.get_output_at(1) for layer in vgg_averagepool.layers \
            if layer.name.endswith('conv1')
        ]
        if self.debug:
            print('created symbolic_conv_outputs')
        # make a big model that outputs multiple layers' outputs
        style_model = Model(inputs=vgg_averagepool.input, outputs=self.symbolic_conv_outputs)
        # calculate the targets that are output at each layer
        self.style_layers_outputs = [K.variable(y) for y in style_model.predict(style_image)]
        # we will assume the weight of the content loss is 1 and only weight the style losses
        style_weights = [0.2, 0.4, 0.3, 0.5, 0.2]
        ## create the total loss which is the sum of content + style loss
        # content loss
        if self.process_var.get() == 'style':
            loss = 0
            # reset to uniform weights to all layers
            style_weights = [1, 1, 1, 1, 1]
        else:
            loss = K.mean(K.square(content_model.output - self.content_target))
        # add weighted style lost
        if self.process_var.get() in {'transfer', 'style'}:
            for w, symbolic, actual in zip(style_weights, self.symbolic_conv_outputs, self.style_layers_outputs):
                # gram_matrix() expects a (H, W, C) as input
                loss += w * self.style_loss(symbolic[0], actual[0])

        if self.debug:
            print('defined loss')
        # create the gradients and loss + grads function
        grads = K.gradients(loss, vgg_averagepool.input)

        get_loss_and_grads = K.function(
            inputs=[vgg_averagepool.input],
            outputs=[loss] + grads
        )

        def get_loss_and_grads_wrapper(x_vec):
            # scipy's minimizer returns function value f(x) and gradients f'(x) simultaneously
            # input to minimizer func must be a np.float64 1-D array
            # gradient must also be a np.float64 1-D array
            loss, gradient = get_loss_and_grads([x_vec.reshape(*batch_shape)])
            return loss.astype(np.float64), gradient.flatten().astype(np.float64)
        if self.debug:
            print('defined wrapper')
        # Initialize optimization
        self.update_status_text("Starting optimization")
        iterations_val = int(self.iterations_val.get())
        self.gui_minimize(get_loss_and_grads_wrapper, iterations_val, batch_shape)
        if self.debug:
            print('end run')

    def gui_minimize(self, fn, epochs, batch_shape):
        """GUI minimize action. Optimizes fn using the L-BFGS algorithm.
        Updates images per iteration"""
        if self.debug:
            print('starting gui_minimize')
        t0 = datetime.datetime.now()
        losses = []
        x = np.random.randn(np.prod(batch_shape))
        for iteration in range(epochs):
            x, loss, _ = fmin_l_bfgs_b(
                func=fn,
                x0=x,
                maxfun=20
            )
            x = np.clip(x, -127, 127)
            # status string
            duration = datetime.datetime.now() - t0
            remaining = epochs*duration/(iteration+1) - duration
            status_str = 'iter={0}, loss={1:.3f}, duration: {2}, remaining: {3}'.format(iteration,
                                                                                                loss,
                                                                                                str(duration).split('.', 2)[0],
                                                                                                str(remaining).split('.', 2)[0])
            self.update_status_text(status_str)
            losses.append(loss)
            # display combined progress (final) image
            resimg = x.copy()
            resimg = resimg.reshape(*batch_shape)
            resimg = self.unpreprocess(resimg)
            # TODO evaluate content image and style image
            # content_img = K.get_value(self.content_target)  # dims?
            self.combined_progress_axis.imshow(self.scale_img(resimg[0,...]))
            self.results_canvas.draw()
        if self.debug:
            print('completed optimization')

    # GUI

    def initialize_gui(self):
        """Create a GUI for style transfer application"""
        self.root = tk.Tk()
        self.root.title("Style Transfer GUI")
        # main frame
        mainframe = tk.Frame(self.root)
        mainframe.grid(row=0, column=0, sticky='nsew')  # TODO add weights to make gui look nicer
        # Content Frame
        content_frame = tk.LabelFrame(mainframe, text="Content")
        content_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        content_load_img_button = tk.Button(content_frame, text="load content image", command=self.load_content_img)
        content_load_img_button.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)
        content_image_frame = tk.Frame(content_frame)
        content_image_frame.grid(row=1, column=0, padx=5, pady=5)
        content_figure = plt.Figure(figsize=(2.5, 2.5), dpi=100)
        self.content_axis = content_figure.add_subplot(111)
        self.content_axis.imshow(self.content_image)
        self.content_axis.axis('off')
        self.content_canvas = FigureCanvasTkAgg(content_figure, master=content_image_frame)  # A tk.DrawingArea.
        self.content_canvas.draw()
        self.content_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        ## Style Frame
        style_frame = tk.LabelFrame(mainframe, text="Style")
        style_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self.style_load_img_button = tk.Button(style_frame, text="load style image", command=self.load_style_img)
        self.style_load_img_button.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)
        self.style_load_img_button.configure(state=tk.DISABLED)  # disable button until content image is loaded
        style_image_frame = tk.Frame(style_frame)
        style_image_frame.grid(row=1, column=0, padx=5, pady=5)
        style_figure = plt.Figure(figsize=(2.5, 2.5), dpi=100)
        self.style_axis = style_figure.add_subplot(111)
        self.style_axis.imshow(self.style_image)
        self.style_axis.axis('off')
        self.style_canvas = FigureCanvasTkAgg(style_figure, master=style_image_frame)  # A tk.DrawingArea.
        self.style_canvas.draw()
        self.style_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        ## Settings Frame
        settings_frame = tk.LabelFrame(mainframe, text="Settings")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        # Transfer button
        self.start_button = tk.Button(settings_frame, text='Optimize', command=self.gui_transfer)
        self.start_button.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.start_button.configure(state=tk.DISABLED)  # disable button until style image is loaded
        # model selection
        model_label = tk.Label(settings_frame, text="Select Model:")
        model_label.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        # Create a transfer model selection variable
        self.model_var = tk.StringVar(self.root)
        self.model_var.set(list(self.models)[0])  # set the default option
        model_select_menu = tk.OptionMenu(settings_frame, self.model_var, *self.models)
        model_select_menu.grid(row=0, column=2, sticky='nsew', pady=5)
        # optimization type selection
        process_label = tk.Label(settings_frame, text="Optimize type")
        process_label.grid(row=0, column=3, sticky='nsew', padx=5, pady=5)
        self.process_var = tk.StringVar(self.root)
        self.process_var.set(list(self.processes)[0])  # set the default option
        process_select_menu = tk.OptionMenu(settings_frame, self.process_var, *self.processes)
        process_select_menu.grid(row=0, column=4, sticky='nsew', pady=5)
        # iteration number
        iterations_label = tk.Label(settings_frame, text="Iterations")
        iterations_label.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        self.iterations_val = tk.Entry(settings_frame)
        self.iterations_val.grid(row=1, column=2, sticky='nsew', pady=5)
        self.iterations_val.delete(0, tk.END)
        self.iterations_val.insert(0, str(self.default_iterations))

        # Cutoff value
        cutoff_label = tk.Label(settings_frame, text="Content cutoff (1-13)")
        cutoff_label.grid(row=1, column=3, sticky='nsew', padx=5, pady=5)
        self.cutoff_val = tk.Entry(settings_frame)
        self.cutoff_val.grid(row=1, column=4, sticky='nsew', pady=5)
        self.cutoff_val.delete(0, tk.END)
        self.cutoff_val.insert(0, str(self.default_cutoff))
        # TODO add weights for combined SSE
        status_frame = tk.LabelFrame(settings_frame, text="Progress")
        status_frame.grid(row=2, column=0, columnspan=5, stick='nsew', padx=5, pady=5)
        status_frame.grid_rowconfigure(0, weight=1)
        status_frame.grid_columnconfigure(0, weight=1)
        scrollbar = tk.Scrollbar(status_frame)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.status_text = tk.Text(status_frame, width=80, height=6,  yscrollcommand=scrollbar.set)
        self.status_text.grid(row=0, column=0, sticky='nsew')
        self.status_text.config(state=tk.DISABLED)
        scrollbar.config(command=self.status_text.yview)
        ## Results Frame
        results_frame = tk.LabelFrame(mainframe, text="Results")
        results_frame.grid(row=0, column=2, rowspan=2, sticky='nsew', padx=5, pady=5)
        results_image_frame = tk.Frame(results_frame)
        results_image_frame.grid(row=0, column=0)
        results_figure = plt.Figure(figsize=(4, 4), dpi=100)
        self.combined_progress_axis = results_figure.add_subplot(111)
        self.combined_progress_axis.imshow(self.style_image)
        self.combined_progress_axis.axis('off')
        self.results_canvas = FigureCanvasTkAgg(results_figure, master=results_image_frame)  # A tk.DrawingArea.
        self.results_canvas.draw()
        self.results_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.root.mainloop()

    def load_content_img(self):
        """Axillary function to load content_img"""
        file_name = filedialog.askopenfilename(initialdir="./",
                                               title="Select content file",
                                               filetypes=(("jpeg files", "*.jp*g"),
                                                            ("all files", "*.*")))
        # load image
        self.content_image = image.load_img(file_name)
        # display image in GUI
        self.content_axis.imshow(self.content_image)
        self.content_canvas.draw()
        self.update_status_text('loaded content image {}'.format(os.path.basename(file_name)))
        # update image size
        self.image_size = self.content_image.size
        # enable load style image button
        self.style_load_img_button.configure(state=tk.NORMAL)

    def load_style_img(self):
        """Axillary function to load style_img"""
        file_name = filedialog.askopenfilename(initialdir="./",
                                               title="Select style file",
                                               filetypes=(("jpeg files", "*.jp*g"),
                                                            ("all files", "*.*")))
        # load and resize image
        target_size = (self.image_size[1], self.image_size[0])
        self.style_image = image.load_img(file_name, target_size=target_size)
        # display image in GUI
        self.style_axis.imshow(self.style_image)
        self.style_canvas.draw()
        self.update_status_text('loaded style image {}'.format(os.path.basename(file_name)))
        # enable start button
        self.start_button.configure(state=tk.NORMAL)

    def update_status_text(self, text=None):
        """Axillary function to update GUI 'status' text
         Parameters
         text   {str} text to display. None to clear terminal
         """
        self.status_text.config(state=tk.NORMAL)
        if text is None:
            self.status_text.delete('1.0', tk.END)
        else:
            # add newline character
            text = text+'\n'
            self.status_text.insert(tk.END, text)
        self.status_text.config(state=tk.DISABLED)

    @staticmethod
    def unpreprocess(img):
        """Reverse VGG preprocessing. Outputs pixel intensities in the range [0,1]
        Values taken from Deep Learning: Advanced Computer Vision Udemy course
        Parameters:
            img     image mapped for VGG
        Output
            RGB mapped image
        """
        img[..., 0] += 103.939
        img[..., 1] += 116.779
        img[..., 2] += 126.68
        img = img[..., ::-1]
        return img

    @staticmethod
    def scale_img(x):
        """Utility function. Scale the image to [0,1].
        May be requred by matplotlib to plot the image
        """
        x = x-x.min()
        x = x/x.max()
        return x

    @staticmethod
    def gram_matrix(img):
        # input is (H, W, C) (C = # feature maps)
        # we first need to convert it to (C, H*W)
        X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))

        # now, calculate the gram matrix
        # gram = XX^T / N
        # the constant is not important since we'll be weighting these
        G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
        return G

    @staticmethod
    def style_loss(y, t):
        """Return SSE distance"""
        return K.mean(K.square(StyleTransfer.gram_matrix(y) - StyleTransfer.gram_matrix(t)))

