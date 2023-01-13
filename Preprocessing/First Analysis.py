def crop_images(file, s_lesions: dict, poly = poly):
  """
  Crop the images to 375x375 without considering where is the kidney
  
  features:
  - file: list of name files
  - s-lesions: dict with the indx and the sum of the lesions each kidney has
  - poly: list of names with the poly mask
  """
  file_name = file.split('.')[0]

  #getting files from the folders_________________
  im = Image.open('images/'+file) #IMAGE
  mask = scipy.io.loadmat('masks/'+file_name) #MASK

  f_poly= file_name
  mask_poly = scipy.io.loadmat('masks_poly/'+f_poly) #MASK_POLY

  im_dim= np.shape(im)[:2] #we get the dimension of the image (and the masks)
  min_s = 375
  
  #getting the number of pixels we need to remove
  h = im_dim[0] - min_s
  w = im_dim[1] - min_s

  left = int(w/2); right = min_s+int(w/2)
  top = int(h/4); bottom = min_s+int(h/4)

 
  ######################################## CROP AND SAVE ####################################################
  #IMAGE
  im2 = im.crop((left, top, right, bottom))
  # im2.save(f"normal_crop/cropped_images/{file}")

  #MASK
  mask2 = mask['mask'][top:bottom, left:right]
  # np.savetxt(f"normal_crop/cropped_masks/{file_name}.txt", mask2, fmt='%s') #fmt = '%s' --> so it saves 0 instead of 0.0000000+00   

  #MASK_POLY
  mask_p2 = mask_poly['mask'][top:bottom, left:right]
  # np.savetxt(f"normal_crop/cropped_masks_poly/{file_name}.txt", mask_p2, fmt='%s')

  ######################################## CROP AND SAVE OF MASK_LESIONS ####################################################
  for lesion in range(s_lesions[file_name]):
    m_lesion = scipy.io.loadmat('lesion_masks/'+file_name+'__'+str(lesion+1)) #LESION_MASK
    m_lesion2 = m_lesion['mask'][top:bottom, left:right]
    # np.savetxt(f"normal_crop/cropped_lesion_masks/{file_name+'__'+str(lesion+1)}.txt", m_lesion2, fmt='%s')

def mask_crop_images(file, s_lesions=s_lesions, poly = poly):
  """
  Crop the images to 375x375 using the mask to consider where is the kidney
  
  features:
  - file: list of name files
  - s-lesions: dict with the indx and the sum of the lesions each kidney has
  - poly: list of names with the poly mask
  """
  file_name = file.split('.')[0]

  #getting files from the folders_________________
  im = Image.open('images/'+file) #IMAGE
  mask = scipy.io.loadmat('masks/'+file_name) #MASK

  f_poly= file_name
  mask_poly = scipy.io.loadmat('masks_poly/'+f_poly) #MASK_POLY

  im_dim= np.shape(im)[:2] #we get the dimension of the image (and the masks)
  min_s = 375
  
  #getting the number of pixels we need to remove
  h = im_dim[0] 
  w = im_dim[1]

  sidx = np.nonzero(mask['mask'])
  
  #what is missing, the border that must be added so that it has 375
  w2= min_s - (sidx[1].max() - sidx[1].min())
  h2= min_s - (sidx[0].max() - sidx[0].min())

  left = np.maximum(sidx[1].min()-int(w2/2),0)
  right = np.minimum(sidx[1].max()+int(w2/2),w)
  if right-left != min_s: #In case it doesn't reach 375 --> guy who has stayed at 374 due to the division
    if right != w: #by default it is added to the right, if it already has the maximum value, it is added to the left
      right += min_s-(right-left)
    else:
      left -= min_s-(right-left)

  top = np.maximum(sidx[0].min()-int(h2/2),0)
  bottom = np.minimum(sidx[0].max()+int(h2/2),h)
  if bottom-top != min_s:
    if bottom != h:
      bottom += min_s-(bottom-top)
    else:
      top -= min_s-(bottom-top)

  ######################################## CROP AND SAVE ####################################################
  #IMAGE
  im2 = im.crop((left, top, right, bottom))
  im2.save(f"mask_crop/mcropped_images/{file}")

  #MASK
  mask2 = mask['mask'][top:bottom,left:right]
  np.savetxt(f"mask_crop/mcropped_masks/{file_name}.txt", mask2, fmt='%s') #fmt = '%s' --> so it saves 0 instead of 0.0000000+00   

  #MASK_POLY
  mask_p2 = mask_poly['mask'][top:bottom,left:right]
  np.savetxt(f"mask_crop/mcropped_masks_poly/{file_name}.txt", mask_p2, fmt='%s')

  ######################################## CROP AND SAVE OF MASK_LESIONS ####################################################
  for lesion in range(s_lesions[file_name]):
    m_lesion = scipy.io.loadmat('lesion_masks/'+file_name+'__'+str(lesion+1)) #LESION_MASK
    m_lesion2 = m_lesion['mask'][top:bottom,left:right]
    np.savetxt(f"mask_crop/mcropped_lesion_masks/{file_name+'__'+str(lesion+1)}.txt", m_lesion2, fmt='%s')
