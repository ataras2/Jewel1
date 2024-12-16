;
; Modified from test_submask.pro - this version does not
;  replicate input into 3-symmetric array
; Function to test for redundancy in sub-arrays of holes
; (this helps speed up the process of searching)

; This version: _submask_ version clones array assuming 3-fold symmetry. Vertex-center for now.
; 		some changes from test_mask_noskirt
;               Assumes non-grid-centered rotation for hole pattern
;
; Future enhancements:
;  switch for centered/non-centered array
;  ability to return partially redundant (better than some cut)

; hacked version noskirt. Ensures that mask is non-redundant AND that
; there are no near-neighbor splodges in fft (to get rid of masks
; where neighbor overlapping power might be a problem)
;
; returns total of all redundancy as a value. Non-redundant should be zero return.

function test_submask_nr,inholes,idim=idim,jdim=jdim,noncent=noncent

s=size(inholes)
nholes=s(2)*3
tha=intarr(2,nholes)
nonred=0
qa=inholes

if(keyword_set(noncent) eq 0) then noncent=0

; Size of parameter space (hole locations) to search:
if (keyword_set(idim) eq 0) then begin
  idim=20
  jdim=20
endif

; Clone qa array onto rotations of -120 and +120 deg (Un-Deleted!)
qb=qa			;qb is -120 deg clone
qb(0,*)=qa(1,*)-qa(0,*)
qb(1,*)=-qa(0,*)
qc=qa			;qc is  120 deg clone
qc(0,*)=-qa(1,*)
qc(1,*)=-qb(0,*)

if(noncent eq 1) then begin
  ; Non-Grid-Centered
  qc(0,*) = qc(0,*)-1
  qb(0,*) = qb(0,*)-1
  qb(1,*) = qb(1,*)-1
endif



t1=[reform(qa(0,*)),reform(qb(0,*)),reform(qc(0,*))]
t2=[reform(qa(1,*)),reform(qb(1,*)),reform(qc(1,*))]
tha(0,*)=t1
tha(1,*)=t2

; now convert from triangular to rectangular grid ...

xyha=intarr(2,nholes)
xyha(1,*)=tha(1,*)
xyha(0,*)=-(2*tha(0,*)-tha(1,*))

; We now have things on a weird rectangular grid. Although
;   this does not reflect the true mask aspect ratio, it should
;   be OK for working out components to see if the mask is 
;   redundant etc etc ...


;Now make up array of expected uv Fourier coverage:
uv=intarr(2,nholes,nholes)
rarr=intarr(8*idim,4*jdim)
for i=0, nholes-1 do begin
  for j=i+1, nholes-1 do begin
    uv(0,i,j)=xyha(0,i)-xyha(0,j) 
    uv(0,j,i)=-uv(0,i,j)
    uv(1,i,j)=xyha(1,i)-xyha(1,j)
    uv(1,j,i)=-uv(1,i,j)

; Transcribe uv onto an array rarr to check for redundancy ...
    rarr(uv(0,i,j)+4*idim,uv(1,i,j)+2*jdim)=    $
         rarr(uv(0,i,j)+4*idim,uv(1,i,j)+2*jdim)+1 
    rarr(uv(0,j,i)+4*idim,uv(1,j,i)+2*jdim)=    $
         rarr(uv(0,j,i)+4*idim,uv(1,j,i)+2*jdim)+1 

  endfor
endfor

; if(max(rarr) le 1) then begin     ;regular NRM criterion
;   r1=rarr+shift(rarr,1,1)
;   r2=rarr+shift(rarr,1,-1)       ; these 3 variables to test for
;   r3=rarr+shift(rarr,2,0)        ;no pspec neighbors
;   if(max([r1,r2,r3]) le 1) then nonred=1 else nonred=max([r1,r2,r3])
; endif

   w_rarr = where(rarr gt 1)
   if((size(w_rarr))[0] gt 0) then nonred = total(rarr(w_rarr)-1)

   r1=rarr+shift(rarr,1,1)
   r2=rarr+shift(rarr,1,-1)       ; these 3 variables to test for
   r3=rarr+shift(rarr,2,0)        ;no pspec neighbors
   r123 = [r1,r2,r3]
   w_r123 = where(r123  gt 1)
   if((size(w_r123))[0] gt 0) then nonred += total(r123(w_r123)-1)

return,nonred

end

