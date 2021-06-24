#!/usr/bin/env python3

'''
Nick's attempt at object oriented programing

Heimdallr wfs code first attempt. 
''' 

##########################################################
#					Import modules
##########################################################

#	Generic
import sys
import numpy as np
import time
import matplotlib.pyplot as plt 							 
from tqdm import tqdm 						# For fancy progress bars

#	More specific tools
from xaosim.shmlib import shm
import xara
import threading							# Enables to launch independent threads



##########################################################
# 			Define generic and usefull functions
##########################################################

# -----------------------------------------------
#	Image Handling
# -----------------------------------------------

def multi_wl_split_frame(img, crop, offsets = None):
	'''---------------------------------------------
	Function to split a single kcam frame into a 
	3 x crop x crop cube of each wl interferogram.

	Parameters:
	-----------
	image: 		a 2D frame from Kcam.
	crop : 		the crop size
	offsets:	the center pts of interferograms
	---------------------------------------------'''
	
	if offsets.any()==None:
		off = np.zeros([3,2]).astype("int")
	else:
		off = offsets.astype("int")

	array = np.zeros([3, crop*2, crop*2 ])
	array[0] = crop_Js(img, crop, offX=off[0,0], offY=off[0,1] ).T
	array[1] = crop_Jl(img, crop, offX=off[1,0], offY=off[1,1] ).T
	array[2] = crop_H(img, crop, offX=off[2,0], offY=off[2,1] ).T
	return array

def crop_Js(image, cropwindow, offX=0, offY=0):
	xx=255+offX
	yy=180+offY
	return image[yy-cropwindow : yy+cropwindow, xx-cropwindow : xx+cropwindow]

def crop_Jl(image, cropwindow, offX=0, offY=0):
	xx=80+offX
	yy=186+offY
	return image[yy-cropwindow : yy+cropwindow, xx-cropwindow : xx+cropwindow]

def crop_H(image, cropwindow, offX=0, offY=0):
	xx=160+offX
	yy=73+offY
	return image[yy-cropwindow : yy+cropwindow, xx-cropwindow : xx+cropwindow]


# -----------------------------------------------
# 	Interferometry
# -----------------------------------------------

def get_CVIS(FTM, imagecube, numholes):
	'''-----------------------------------------------
	Extracts CVIS values from imagecube

	Parameters:
	-------------
	FTM:		The Fourier-Transfer Matrix (from make_FTM() )
	imagecube:	The cube of interferogram images to be analysed
	numholes:	The number of mask-holes/apertures/telescopes

	-----------------------------------------------'''

	cvis=np.zeros((FTM.shape[0], FTM.shape[1]), dtype=complex)
	for w in range(cvis.shape[0]):
		cvis[w] = FTM[w].dot(imagecube[w].flatten()) * (numholes/imagecube[w].sum())
	return cvis

def get_CVIS_D(FTM, image, numholes):
	'''-----------------------------------------------
	Extracts CVIS values from imagecube

	Parameters:
	-------------
	FTM:		The Fourier-Transfer Matrix (from make_FTM() )
	image:		The interferogram image to be analysed
	numholes:	The number of mask-holes/apertures/telescopes

	-----------------------------------------------'''

	cvis=np.zeros((FTM.shape[0]), dtype=complex)
	cvis = FTM.dot(image.flatten()) * (numholes/image.sum())
	
	return cvis

def visphase_to_complex(amp, phase):
	'''
	Small function to convert Amplitude and
	Phase into a complex visibility.
	'''
	return amp*np.exp( 1j*phase)


def Idealfringe_array(OPD, wl):
	'''
	Makes a complex visibility with amp=1
	given a particulat OPD and wavelength.
	'''
	return np.exp( 1j*((2*np.pi*OPD)/wl) )


def baseline2pupil(pinv, base_phs_or_pist):
	'''
	Function that maps baselines into pupil-plane
	apertures using the provided PINV.
	'''
	return  np.append(0,  pinv.dot(base_phs_or_pist) )


def calibratefringe(cvis, cvis0):
	'''
	calibrates a given complex visibility by the provided 
	zero-point. 
	Divides the amplitude, and subtracts the phase.
	'''
	return (np.abs(cvis)/np.abs(cvis0))*np.exp( 1j*(np.angle(cvis)-np.angle(cvis0)) )



# =======================================================
# =======================================================

class WFS():
	#===============================================
	def __init__(self, shmf="/dev/shm/kcam.im.shm", smh_dark="/dev/shm/kcam_dark.im.shm",
		shm_wfs = "/dev/shm/heim_wfs.00.shm", shm_wfs_dense = "/dev/shm/heim_wfs.01.shm", model_path = "heim_singseg_mid_cor.fits.gz", 
		densemodel_path="heim_singseg_mid_TT.fits.gz", shm_guiarrays="/dev/shm/guiarrays.im.shm" ):
		'''-----------------------------------------
		Defult WFS class constructor.

		Parameters: 
		----------
		- shmf:	shared memory structure where to get images
		- smh_dark : path to dark
		- model_path : path and name of the Pupil Model
		-----------------------------------------'''

		
		# ---------------------------------------------------------------------
		# Hard coded variables
		# --------------------------------------------------------------------
		self.ntel = 4										# Num of Telescopes
		self.wavs = np.array([1.09e-6, 1.34e-6, 1.60e-6])	# cent WL of Chnls
		self.pixsz = 24										# cam pix size in um
		self.lens = 750										# focal len to cam
		self.pscale = 200*self.pixsz/self.lens 				# platescale

		self.im_wl_coord = np.zeros((3,2))					# coords of the interferograms
		self.im_crop = 60									# interferogram crop window

		self.s_crop = 66  					# Number of singular values to crop for WFS
		self.apperhole = 19   				# number of apertures sampeling each telescope

		# NCP error corrector
		self.ncpe_frames = 100
		self.ncpe_fps = 50

		# Correlator
		self.c_start = -5					# start point of correlator in um
		self.c_stop = 5						# end point of correlator in um
		self.c_rez = 1e-3 					# correlator resolution in um


		self.mod =["Js", "Jl", "H"]

		self.OPD2phs = 2*np.pi/self.wavs

		# ----------------------------------------
		# Shared memory structures and variables:
		# ----------------------------------------

		# KCAM
		self.shm_im = shm(shmf)					# the shared memory image
		self.shm_drk= shm(smh_dark)
		self.dark = self.shm_drk.get_data()[1:,:].astype("float")

		self.iszx = self.shm_im.mtdata['x']		# image size X
		self.iszy = self.shm_im.mtdata['y']		# image size Y
		self.im_cnt = self.shm_im.get_counter()	# image counter

		# Interferograms (for GUI)
		self.guiarrays = np.zeros([3, self.im_crop*2, self.im_crop*2 ], dtype="float32")
		self.shm_guiarr = shm(fname=shm_guiarrays, data=self.guiarrays)	

		# WFS Outputs
		self.shm_pist = shm(fname=shm_wfs, data= np.zeros((self.ntel,1), dtype="float32"))
		self.pistons_cor = np.zeros((self.ntel))						# measured pistons by correlator
		self.pistons_inter = np.zeros((self.wavs.shape[0],self.ntel))	# measured pistons by interferometer


		self.shm_pist_dense = shm(fname=shm_wfs_dense, data= np.zeros((self.ntel,3), dtype="float32"))
		# self.pistons_cor_dense = np.zeros((self.ntel))
		self.PTT = np.zeros((self.ntel,3))



		self.time_0 = 0							# Time when loop begins 
		self.frame_time = 0						# Time when frame is taken
		self.time_log = []						# Log of all the times
		self.pist0_log = []						# Log of T0 pistons
		self.pist1_log = []						# Log of T1 pistons 
		self.pist2_log = []						# ' '    T2  '  '
		self.pist3_log = []						# ' '    T3  '  '
		self.log_len = 1000						# lenght of the log 

		
		self.kpomodel = xara.KPO(model_path)	# Load of the KPI model
		self.model_path = model_path			# Link to the KPI model name

		self.kpomodel_dense = xara.KPO(densemodel_path)	# Load of the dense KPI model
		self.model_path_dense = densemodel_path


		##### Empty arrays
		self.CVIS_0 = np.zeros([self.wavs.shape[0], 6], dtype=complex)
		self.FTM = None
		self.IFM = None
		self.OPDs = None
		self.PINV = None

		self.nbaselines_D =self.kpomodel_dense.kpi.nbuv
		self.nholes_D =self.kpomodel_dense.kpi.nbap
		self.CVIS_0_D = np.zeros((self.kpomodel_dense.kpi.TFM.shape[0]), dtype=complex)
		self.FTM_D = None
		self.IFM_D = None
		self.OPDs_D = None
		self.PINV_D = None

		self.pistons_D = np.zeros((self.nholes_D))

		self.diag_correlator = None
		self.delay=0
		self.nframes=None


		# ----------------------------------------
		# control parts
		# ----------------------------------------

		self.keepgoing = False 					# Close-Loop Flag

		

	# ===============================================================

	def make_FTM(self,):
		'''----------------------------------------------------
		Makes the Fourier-Transfer Matrix. This is used to 
		obtain Complex Visibilities from a datacube.

		Updates FTM variable with new matrix. 
		----------------------------------------------------'''
		
		img = self.shm_im.get_data()[1:,:] - self.dark
		data = multi_wl_split_frame(img, self.im_crop, offsets= self.im_wl_coord)

		models={}

		for w in range(self.wavs.shape[0]):
			models['kpo_'+str(self.mod[w])] = xara.KPO(self.model_path)
		
		self.CVIS_0 = np.zeros([self.wavs.shape[0], models["kpo_Js"].kpi.TFM.shape[0]], dtype=complex)

		for w in range(self.wavs.shape[0]):
			models['kpo_' +str(self.mod[w])].extract_KPD_single_frame(data[w], self.pscale, self.wavs[w], recenter=False)
			self.CVIS_0[w,:] = models['kpo_'+str(self.mod[w])].CVIS[0][0]

		for w in range(self.wavs.shape[0]):
			models['FTM_' + str(self.mod[w])] = models['kpo_' + str(self.mod[w])].FF
		
		self.FTM = np.stack([models["FTM_Js"], models["FTM_Jl"], models["FTM_H"]], axis=0)
	
	# ===============================================================

	def make_FTM_dense(self,):
		'''----------------------------------------------------
		Makes the Fourier-Transfer Matrix. This is used to 
		obtain Complex Visibilities from a datacube.
		Only For the TIP-TILT Model
		H-BAND Only!

		Updates FTM variable with new matrix. 
		----------------------------------------------------'''
		
		img = self.shm_im.get_data()[1:,:] - self.dark
		data = multi_wl_split_frame(img, self.im_crop, offsets= self.im_wl_coord)

		self.CVIS_0_D = np.zeros((self.nbaselines_D), dtype=complex)
		tmpmod = xara.KPO(self.model_path_dense)

		tmpmod.extract_KPD_single_frame(data[2], self.pscale, self.wavs[2], recenter=False)

		self.CVIS_0_D = tmpmod.CVIS[0][0]
		self.FTM_D = tmpmod.FF

	# ===============================================================

	def get_NCP_offsets(self,):
		'''----------------------------------------------------
		Measures non-common-path DC offsets between the 3 Wls.
		Updates the zero-point self.CVIS_0 variable.
		
		*Note* - The self.FTM must be made beforehand using
				make_FTM() function.

		Uses:
		self.ncpe_fps : 	the speed in fps of the averaging
		self.ncpe_frames:	the number of frames to be averaged
		---------------------------------------------------'''
		if self.FTM is not None:
			if self.FTM_D is not None:

				print("Measuring NCP Offsets:")
				cvis_ncp_log = np.zeros((self.ncpe_frames, self.CVIS_0.shape[0], self.CVIS_0.shape[1]), dtype=complex) 
				cvis_ncp_log_D = np.zeros((self.ncpe_frames, self.CVIS_0_D.shape[0]), dtype=complex) 

				for f in tqdm(range(self.ncpe_frames)):
					img = self.shm_im.get_data()[1:,:] - self.dark
					data = multi_wl_split_frame(img, self.im_crop, offsets= self.im_wl_coord)
					cvis_ncp_log[f] = get_CVIS(self.FTM, data, self.ntel)
					cvis_ncp_log_D[f] = get_CVIS_D(self.FTM_D, data[2], self.nholes_D)

					time.sleep(1/self.ncpe_fps)
				self.CVIS_0 = visphase_to_complex(np.median(np.abs(cvis_ncp_log), axis=0), np.median(np.angle(cvis_ncp_log), axis=0))
				self.CVIS_0_D = visphase_to_complex(np.median(np.abs(cvis_ncp_log_D), axis=0), np.median(np.angle(cvis_ncp_log_D), axis=0))
				print("CVIS_0's are updated")
			else:
				print("No FTM_D detected.. plz run 'make_FTM_dense()' funct. first ")
		else:
			print("No FTM detected.. plz run 'make_FTM()' funct. first ")

	# ===============================================================

	def make_IFM(self,):
		'''---------------------------------------------------------
		makes the Ideal-Fringe Matrix and stores it. This is used
		in the correlator function.
		
		Updates IFM variable
		---------------------------------------------------------'''
		self.OPDs = np.arange(self.c_start, self.c_stop, self.c_rez)*1e-6
		opds_mesh,wave_mesh = np.meshgrid(self.OPDs, self.wavs)
		
		IFM_s = Idealfringe_array(opds_mesh,wave_mesh)
		self.IFM = np.repeat(IFM_s.T[ :, :, np.newaxis], self.kpomodel.kpi.nbuv, axis=2)

	# ===============================================================

	def OPD_correlator(self, complex_visibilities, diagnostic = False):
		'''-----------------------------------------------------------
		Correlator function takes the complex visibilities provided
		and calculates the Pistons for each aperture.

		Parameters:
		-----------
		complex_visibilities: 	a 3x6 array of complex visibilities

		-----------------------------------------------------------'''
		if self.IFM is not None:
			if self.PINV is not None:

				self.pistons_cor = baseline2pupil(self.PINV, self.OPDs[np.argmax(np.sum(self.IFM*complex_visibilities, axis=1).real, axis=0) ])
				
				if diagnostic:
					self.diag_correlator = self.IFM * complex_visibilities

			else:
				print("PINV not found. Plz run 'make_PINV()' function")
		else:
			print("IFM not found. Plz run 'make_IFM()' function.")

	# ===============================================================

	def WFS_cor(self,):
		'''---------------------------------------------------------
		The main WFS function

		Will take a frame from the given shared memory path, 
		subtract the dark, calculate the CVIS, use correlator to get
		the OPDs, then place them in the pistons_cor variable. 
		Will also keep a timestamp relative to the begining of the 
		loop.
		---------------------------------------------------------'''

		im = self.shm_im.get_data()[1:,:] - self.dark
		data = multi_wl_split_frame(im, self.im_crop, offsets = self.im_wl_coord)
		self.guiarrays =data.astype(np.float32)
		
		CVIS = get_CVIS(self.FTM, data, self.ntel)
		CVIS_norm = calibratefringe(CVIS, self.CVIS_0)
		self.OPD_correlator(CVIS_norm) 
		self.frame_time = time.time() - self.time_0

	# ===============================================================

	def WFS_inter(self,):
		'''
		Same as WFS_cor() but only takes 1 WL chanel not all. Much faster.
		'''
		im = self.shm_im.get_data()[1:,:] - self.dark
		data = multi_wl_split_frame(im, self.im_crop, offsets = self.im_wl_coord)

		CVIS = get_CVIS(self.FTM, data, self.ntel)
		CVIS_norm = calibratefringe(CVIS, self.CVIS_0)

		pistons = np.zeros_like(self.pistons_inter)

		for w in range(self.wavs.shape[0]):
			pistons[w] =  baseline2pupil(self.PINV, np.angle(CVIS_norm[w])) /self.OPD2phs[w]

		self.pistons_inter = pistons

	# ===============================================================

	def WFS_dense(self,):
		'''
		Same as WFS_inter() but for TTmodel
		'''
		im = self.shm_im.get_data()[1:,:] - self.dark
		data = multi_wl_split_frame(im, self.im_crop, offsets = self.im_wl_coord)
		self.guiarrays =data.astype(np.float32)

		CVIS = get_CVIS_D(self.FTM_D, data[2], self.nholes_D)
		CVIS_norm = calibratefringe(CVIS, self.CVIS_0_D)

		pistons = np.zeros_like(self.pistons_D)

		pistons = baseline2pupil(self.PINV_D, np.angle(CVIS_norm)) /self.OPD2phs[2]

		self.pistons_D = pistons

	# ===============================================================

	def make_PINV(self, s_crop=None):
		'''
		Makes the Pseudo Inverse matrix that maps baselines to apertures in 
		the pupil-plane. 
		'''
		U, S, Vt = np.linalg.svd(self.kpomodel.kpi.TFM, full_matrices=0)
		S_inverse = 1 / S

		if s_crop is not None:
			nkeep = s_crop  
			S_inverse[nkeep:] = 0.0
			print("Keeping "+str(nkeep)+" Singular values out of " +str(len(S_inverse)))

		self.PINV = Vt.T.dot(np.diag(S_inverse)).dot(U.T)

	# ===============================================================

	def make_PINV_D(self, s_crop=None):
		'''
		Makes the Pseudo Inverse matrix that maps baselines to apertures in 
		the pupil-plane. 
		'''
		U, S, Vt = np.linalg.svd(self.kpomodel_dense.kpi.TFM, full_matrices=0)
		S_inverse = 1 / S

		if s_crop is not None:
			nkeep = s_crop  
			S_inverse[nkeep:] = 0.0
			print("Keeping "+str(nkeep)+" Singular values out of " +str(len(S_inverse)))

		self.PINV_D = Vt.T.dot(np.diag(S_inverse)).dot(U.T)
	
	# ===============================================================

	def calc_PTT(self, ):
		opd_mean = np.zeros((self.ntel))
		for tel in range(self.ntel):
			opd_mean[tel] = np.mean( self.pistons_D[(tel*self.apperhole) : self.apperhole+(tel*self.apperhole)], axis=0)

		for tel in range(1, self.ntel):
			opd_mean[tel] -= opd_mean[0]
		opd_mean[0] -= opd_mean[0]

		self.PTT[:,0] = -opd_mean *1e6 *1e6


		for tel in range(self.ntel):
			X = self.kpomodel_dense.kpi.VAC[(tel*self.apperhole):self.apperhole+(tel*self.apperhole), 0]
			Y = self.kpomodel_dense.kpi.VAC[(tel*self.apperhole):self.apperhole+(tel*self.apperhole), 1]

			singhol= -(self.pistons_D[(tel*self.apperhole):self.apperhole+(tel*self.apperhole)] -np.mean(self.pistons_D[(tel*self.apperhole):self.apperhole+(tel*self.apperhole)]))
			self.PTT[tel,1] = (np.dot(singhol,X)) *0.5*1e6  #/(np.dot(X,X))
			self.PTT[tel,2] = (np.dot(singhol,Y)) *0.5*1e6 #/(np.dot(Y,Y))

		# Mask centres located at index: self.kpomodel_dense.kpi.VAC[0,19,38,57]

	# ===============================================================
	
	def update_interferogram_positions(self,):
		'''-------------------------------------------------------
		Updates the center coords of the 3 interferograms on kcam
		-------------------------------------------------------'''

		ff= self.shm_im.get_data()[1:,:] - self.dark
		im_cube = multi_wl_split_frame(ff, self.im_crop, offsets = self.im_wl_coord)
		im_centres = np.zeros((self.wavs.shape[0], 2))
		for w in range (self.wavs.shape[0]):
			im_centres[w] = np.unravel_index(np.argmax(im_cube[w], axis=None), im_cube[w].shape)
		self.im_wl_coord = ( im_centres - self.im_crop)

	# ===============================================================

	def update_log(self,):
		self.time_log.append(self.frame_time)
		self.pist0_log.append(self.pistons_cor[0])
		self.pist1_log.append(self.pistons_cor[1])
		self.pist2_log.append(self.pistons_cor[2])
		self.pist3_log.append(self.pistons_cor[3])
		if len(self.pist1_log) > self.log_len :
			self.pist0_log.pop(0)
			self.pist1_log.pop(0)
			self.pist2_log.pop(0)
			self.pist3_log.pop(0)
			self.time_log.pop(0)

	# ===============================================================

	def update_shm(self, denseonly=False):
		'''
		Updates the shared memory structure of WFS results.
		'''
		if denseonly:
			self.shm_pist.set_data((self.PTT[:,0]*1e-6).astype("float32")*1e-6)
			self.shm_guiarr.set_data(self.guiarrays)
			self.shm_pist_dense.set_data(self.PTT.astype("float32"))

		else:
			self.shm_pist.set_data(self.pistons_cor.astype("float32"))
			self.shm_guiarr.set_data(self.guiarrays)
			self.shm_pist_dense.set_data(self.PTT.astype("float32"))


	# ===============================================================

	def plot_log_cor(self,):
		ms=1
		mx= np.max([np.max(np.abs(self.pist1_log)),
			np.max(np.abs(self.pist2_log)), 
			np.max(np.abs(self.pist3_log))])*1e9 +10

		plt.ion()
		plt.figure()
		plt.plot(np.array(self.time_log)*ms, np.array(self.pist0_log)*1e9, 'k-')
		plt.plot(np.array(self.time_log)*ms, np.array(self.pist1_log)*1e9, '-')
		plt.plot(np.array(self.time_log)*ms, np.array(self.pist2_log)*1e9, '-')
		plt.plot(np.array(self.time_log)*ms, np.array(self.pist3_log)*1e9, '-')
		plt.title(" WFS Output ")
		plt.xlabel("Time (s)")
		plt.ylabel("Relatinve Piston (nm)")
		plt.legend(["UT-1", "UT-2", "UT-3", "UT-4"])
		plt.axis([None, None, -mx, mx])
		plt.show()

	# ===============================================================

	def loop(self, delay = 0, nframes = None):
		'''------------------------------------------------------------
		Loop function. 

		Parameters:
		-----------
		frames: 	if None will run loop untill stop() is called.
					if number will run for given number of frames.

		------------------------------------------------------------'''
		self.time_0 = time.time()
		
		if nframes is not None:
			print(" Loop started for "+str(nframes)+" frames")
			for f in tqdm(range(nframes)):
				self.WFS_cor()
				self.update_shm()
				self.update_log()
				time.sleep(delay) 
			print("Loop Done")
			self.stop()
		
		else:
			print("Loop started. Use '.stop() to stop the loop")
			self.keepgoing = True
			while self.keepgoing:
				self.WFS_cor()
				self.update_shm()
				self.update_log()
				time.sleep(delay)  

	# ===============================================================

	def loop_D(self, delay = 0, nframes = None):
		'''------------------------------------------------------------
		Loop function. 

		Parameters:
		-----------
		frames: 	if None will run loop untill stop() is called.
					if number will run for given number of frames.

		------------------------------------------------------------'''
		self.time_0 = time.time()
		
		if nframes is not None:
			print(" Loop started for "+str(nframes)+" frames")
			for f in tqdm(range(nframes)):
				self.WFS_dense()
				self.calc_PTT()
				self.update_shm(denseonly=True)
				
				time.sleep(delay) 
			print("Loop Done")
			self.stop()
		
		else:
			print("Loop started. Use '.stop() to stop the loop")
			self.keepgoing = True
			while self.keepgoing:
				self.WFS_dense()
				self.calc_PTT()
				self.update_shm(denseonly=True)
				time.sleep(delay)  

	# ===============================================================

	def saveloop(self, wait4img=True, nframes=1):
		'''
		Function that will run the MW-WFS for a set number of frames
		then save the output in an numpy array.

		if wait4img is true, then the loop will only update when it sees 
		a new frame loaded in the shm.  
		
		'''
		savpth= ""

		# User Input to change filename 

		print("")
		print("ENTER the filename for save file (no .npy at end): ")

		savnm = str(input())

		print("")
		print("Thanks!")


		print(" Loop started for "+str(nframes)+" frames")
		savearray=np.zeros((100,4,3))
		if wait4img:
			self.im_cnt = self.shm_im.get_counter()
			for f in tqdm(range(nframes)):
				imgindx= self.shm_im.get_data(check=self.im_cnt)[0,0]
				self.WFS_cor()
				self.im_cnt = self.shm_im.get_counter()
				# self.update_shm()
				savearray[imgindx,:,0]=self.pistons_cor.astype("float32")

			# savearray=np.append(savearray, imgindx, axis=0)
			np.save(savpth+savnm+".npy", savearray)


		#--------------------------------------------------------------------
		#--------------------------------------------------------------------

		# savearray=np.zeros_like(np.expand_dims(self.pistons_cor.astype("float32"),axis=1))
		# imgindx=np.zeros((1,nframes+1))
		# if wait4img:
		# 	self.im_cnt = self.shm_im.get_counter()
		# 	for f in tqdm(range(nframes)):
		# 		imgindx[0,f+1]= self.shm_im.get_data(check=self.im_cnt)[0,0]
		# 		self.WFS_cor()
		# 		self.im_cnt = self.shm_im.get_counter()
		# 		self.update_shm()
		# 		tmp=np.expand_dims(self.pistons_cor.astype("float32"),axis=1)
		# 		savearray=np.append(savearray,tmp , axis=1)
		# 		# self.im_cnt = self.shm_im.get_counter()

		# 	savearray=np.append(savearray, imgindx, axis=0)
		# 	np.save(savnm, savearray)


		# ===============================================================

	# ===============================================================

	def saveloop_D(self, wait4img=True, nframes=1):
		'''
		Function that will run the MW-WFS for a set number of frames
		then save the output in an numpy array.

		if wait4img is true, then the loop will only update when it sees 
		a new frame loaded in the shm.  
		
		'''

		savpth= ""

		# User Input to change filename 

		print("")
		print("ENTER the filename for save file (no .npy at end): ")

		savnm = str(input())

		print("")
		print("Thanks!")



		print(" Loop started for "+str(nframes)+" frames")
		savearray=np.zeros((100,4,3))
		if wait4img:
			self.im_cnt = self.shm_im.get_counter()
			for f in tqdm(range(nframes)):
				
				imgindx= self.shm_im.get_data(check=self.im_cnt)[0,0]
				self.WFS_dense()
				self.calc_PTT()
				self.im_cnt = self.shm_im.get_counter()

				savearray[imgindx,:,:]=self.PTT.astype("float32")

			
			np.save(savpth+savnm+".npy", savearray)

		#---------------------------------------------------------------------------------------
		# print(" Loop started for "+str(nframes)+" frames")
		# savearray=np.zeros_like(np.expand_dims(self.PTT.astype("float32"),axis=2))
		# imgindx=np.zeros((1,nframes+1))
		# if wait4img:
		# 	self.im_cnt = self.shm_im.get_counter()
		# 	for f in tqdm(range(nframes)):
		# 		imgindx[0,f+1]= self.shm_im.get_data(check=self.im_cnt)[0,0]
		# 		self.WFS_dense()
		# 		self.calc_PTT()
		# 		self.im_cnt = self.shm_im.get_counter()
		# 		self.update_shm()
		# 		tmp=np.expand_dims(self.PTT.astype("float32"),axis=2)
		# 		savearray=np.append(savearray,tmp , axis=2)

		# 	savearray=np.append(savearray, imgindx, axis=0)
		# 	np.save("test_output1.npy", savearray)

	# ===============================================================

	def initialize(self,):
		print("Initializing the WFS")
		
		self.update_interferogram_positions()
		self.make_FTM()
		self.make_FTM_dense()
		self.get_NCP_offsets()
		self.make_PINV()
		self.make_PINV_D(s_crop=self.s_crop)
		self.make_IFM()

	# ===============================================================

	def start(self, delay=0, nframes = None):
		'''-----------------------------------
		Starts the loop, pases the following
		parameters to the loop function:

		Parameters:
		-----------
		delay: 		a forced delay in s
		nframes:	optional instead of endless

		-----------------------------------'''

		self.delay = delay
		self.nframes = nframes

		if not self.keepgoing:
			self.keepgoing = True
			t = threading.Thread( target = self.loop, args=(self.delay, self.nframes))
			t.start()
			print(" WFS loop is running...")


		else:
			print(" WFS Loop is already running")
	
	# ===============================================================
	
	def start_D(self, delay=0, nframes = None):
		'''-----------------------------------
		Starts the loop, pases the following
		parameters to the loop function:

		Parameters:
		-----------
		delay: 		a forced delay in s
		nframes:	optional instead of endless

		-----------------------------------'''

		self.delay = delay
		self.nframes = nframes

		if not self.keepgoing:
			self.keepgoing = True
			t = threading.Thread( target = self.loop_D, args=(self.delay, self.nframes))
			t.start()
			print(" WFS D loop is running...")


		else:
			print(" WFS D Loop is already running")

	# ===============================================================

	def stop(self,):
		'''--------------------------------------
		simple high-level accessor to interupt the
		thread of the WFS loop
		--------------------------------------'''
		if self.keepgoing:
			print("The WFS loop was stopped")
			self.keepgoing=False
		else:
			print("The loop was not running...")


# ====================================================
# =====================================================

if __name__ == "__main__":
	mon = WFS(shmf="/dev/shm/kcam.im.shm" , smh_dark="/dev/shm/kcam_dark.im.shm")
	mon.initialize()





