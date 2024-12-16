; Restore solution

infile='find46_4x7h3x6h.dat'
nholes=7
nmasks=7

lastbig = 4 ; masks after this one are (n-1) segment

idim=9
jdim=9
noncent=0 ; set to 0 for centered grid


;hexcell     = 1.8*sqrt(3)/2 +.005 ;m
hexcell = .9
holesize=0.4 ; meters diameter
primary_dia =  7.92 ;m
sec_dia     = 2.285 ;m


nl=nlines(infile)
nl=nl-1                       ; don't count header line
nsol= nl/(nholes + 1)                    ; 3 lines per data entry
maskxy=fltarr(nsol,nmasks,nholes)  ; 
metrics=intarr(2,nsol)

dd=intarr(2)
ss=' '

ff=intarr(nmasks)

openr,unit,infile,/get
  readf,unit,ss          ; remove header line
  for i=0,nsol-1 do begin
     readf,unit,ss
     line = strsplit(ss,' ',/extract)
     reads,line,dd & metrics[*,i]=dd
     for h=0,nholes-1 do begin
        readf,unit,ss
        line = strsplit(ss,' ',/extract)
        reads,line,ff
        maskxy[i,*,h] = ff
     endfor
  endfor
close,unit

sortr=sort(metrics[1,*])
metrics=metrics[*,sortr]
maskxy=maskxy[sortr,*,*]


; Clock through various solutions below....

sn=-1
startover:
!p.multi=[0,2,1]
print,'Input Solution Number'
read,ss
if(ss eq '') then sn=sn+1 else reads,ss,sn
if(sn ge nsol) then goto,endall

print,'Solution ',metrics[0,sn],' with total redundancy = ',metrics[1,sn]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Plot out the pupil
;hexcell=1.0 ; 
m_sep=0.0
;noncent = 0

edgetol=hexcell*0.1 + primary_dia*0.01
holesize=hexcell*0.3

plot,[0,0],psym=4,xr=[-4.5,4.5],yr=[-4.5,4.5],/nodata,/isotropic,xs=1,ys=1;,background=255 ;,xs=4,ys=4

tvcircle,primary_dia/2.,[-1*m_sep,m_sep]/2.,[0,0],/data
tvcircle,sec_dia/2.,[-1*m_sep,m_sep]/2.,[0,0],/data
;tvcircle,masksize,[0],[0],/data
oplot,[0,-1*m_sep,m_sep]/2.,[0,0,0],psym=1

ix=(findgen(201)-100.) 
ii=(fltarr(201)+1.)  * hexcell

gridxy=[ [reform( (ii##ix+ix##ii/2.)[*])], [reform( (ix##ii*sqrt(3)/2.)[*])]]
; rightdist=sqrt( (ii##ix+ix##ii/2.-m_sep/2.0)^2 + (ix##ii*sqrt(3)/2.)^2)
; leftdist=sqrt( (ii##ix+ix##ii/2.+m_sep/2.0)^2 + (ix##ii*sqrt(3)/2.)^2)

ix_n = (-1.*(findgen(idim)-idim/2))
ixi_n = (fltarr(idim)+1.) * hexcell
gridxy_n=[ [reform( (ixi_n##ix_n-ix_n##ixi_n/2.)[*])], [reform( (ix_n##ixi_n*sqrt(3)/2.)[*])]]
;gridxy_n = reverse(gridxy_n,1)
;gridxy_n = reverse(gridxy_n,2)
; rightdist_n=sqrt( (ixi_n##ix_n+ix_n##ixi_n/2.-m_sep/2.0)^2 + (ix_n##ixi_n*sqrt(3)/2.)^2)
; leftdist_n=sqrt( (ii##ix+ix##ii/2.+m_sep/2.0)^2 + (ix##ii*sqrt(3)/2.)^2)



if(noncent) then begin
   gridxy[*,0]=gridxy[*,0] + hexcell/2
   gridxy[*,1]=gridxy[*,1] + (hexcell/2 * 0.634625)
;    rightdist=sqrt( (ii##ix+ix##ii/2.+hexcell/2-m_sep/2.0)^2 + (ix##ii*sqrt(3)/2.-(hexcell/2*0.634625))^2)
;    leftdist=sqrt( (ii##ix+ix##ii/2.+hexcell/2+m_sep/2.0)^2 + (ix##ii*sqrt(3)/2.- (hexcell/2*0.634625))^2)

   gridxy_n[*,0]=gridxy_n[*,0] - hexcell/2
   gridxy_n[*,1]=gridxy_n[*,1] - (hexcell/2 * 0.634625)
endif

;oplot,ii##ix+ix##ii/2.,ix##ii*sqrt(3)/2.,psym=3
oplot,gridxy[*,0],gridxy[*,1],psym=3


color_vect=[255,240,210,190,130,50,30]

; try some hexagons instead.

x_hexsize= .38 ; holesize ;*pscale/2.0
y_hexsize=x_hexsize*2./sqrt(3)

x_hexgrid=[-1.0,-1.0,0.0,1.0,1.0,0.0,-1.0]  * x_hexsize
y_hexgrid=[-0.5,0.5,1.0,0.5,-0.5,-1.0,-0.5] * y_hexsize
xyhex=[[x_hexgrid],[y_hexgrid]]

for mm=0,nmasks-1 do begin
 for h=0,nholes-1 do begin
    if(( mm ge lastbig ) and (h eq nholes-1)) then goto,skipit
    oplot,xyhex[*,0]+replicate(gridxy_n[reform(maskxy(sn,mm,h)),0],7),xyhex[*,1]+replicate(gridxy_n[reform(maskxy(sn,mm,h)),1],7),color=color_vect(mm),thick=4
    skipit:
  endfor
endfor

dopspec:
; Now do the power spectrum ...

;stop

;wset,1
;window,1

;nirvana_ptscl=7./1000  ; mas/pixel (just suppose)
;plate_scale = (180.*(60.^2)/3.14159265) / nirvana_ptscl  ;%% pixels per radian
;mask_scale  = 1016./(6.625 * 1000.)                               ;%% mm per pixel
;lambda=2.25
;wav=lambda*1.0e-6
;res=3
holes=18./2
;holes=9./2

hsize=holes

; work out spatial freq for one meter baseline
; rad2mas(wav/1.)/7 (mas/px) = 66 pixels.
; We want this baseline say at 5 pix from the center
;   for dixplay, then we multiply "expect0" by 5 ...

; now make up expected pwr spectrum...
;    xyuv=xyholes/10 + 500
;    xyuv=reform(xyuv/res)
;    xy0=round(xyuv*256*mask_scale/(wav*plate_scale))


  ; Firstly get Central Spike ...
  ;  tmp=fltarr(512,512)
  ;  cookiecutter,tmp,expect0(0,0,0),expect0(1,0,0),holes,1.0
    im=fltarr(512,512)

hxp = 380  
for  mm = 0,nmasks-1 do begin 

xy0 = gridxy_n[reform(maskxy(sn,mm,*)),*]
  
  ;Now make up array of expected locations of spots in FFT ...
    expect0=fltarr(2,nholes,nholes)
    for i=0, nholes-1 do begin
      for j=i+1, nholes-1 do begin
        ;expect0(0,i,j)=round( (xy0(i,0)-xy0(j,0)) )
        expect0(0,i,j)=( (xy0(i,0)-xy0(j,0)) )
        expect0(0,j,i)=-expect0(0,i,j)
        ;expect0(1,i,j)=round( (xy0(i,1)-xy0(j,1)) )
        expect0(1,i,j)=( (xy0(i,1)-xy0(j,1)) )
        expect0(1,j,i)=-expect0(1,i,j)
      endfor
    endfor
    
    expect0=expect0*12+256
  


  ; Now cycle through getting pairs of spots i,j & j,i
    for i=0, nholes-1 do begin
      for j=i+1, nholes-1 do begin
           if(( mm ge lastbig) and (j eq nholes-1)) then goto,skipit2
         tmp=fltarr(512,512)
         cookiecutter,tmp,expect0(0,i,j),expect0(1,i,j),hsize,1.0
         if(mm lt 6) then begin  ; hack for 7th pattern
               im=im+shift(tmp,xyhex[mm,0]*hxp,xyhex[mm,1]*hxp)
         endif else im=im +tmp
         tmp=fltarr(512,512)
         cookiecutter,tmp,expect0(0,j,i),expect0(1,j,i),hsize,1.0
         if(mm lt 6) then begin  ; hack for 7th pattern
              im=im+shift(tmp,xyhex[mm,0]*hxp,xyhex[mm,1]*hxp)
         endif else im=im +tmp
        skipit2:
       endfor
    endfor
    ;print,'recalc merit =',total( (im(where(im gt 1)))^2)
    im[0,0]=max(im)*1.05

endfor
    image_cont,im,/nocont,/asp
    for mm=0,nmasks-2 do begin
       ;oplot,[256+xyhex[mm,0]*hxp*.42,256+xyhex[mm,0]*hxp*.45],[256+xyhex[mm,1]*hxp*.42,256+xyhex[mm,1]*hxp*.45],color=color_vect(mm),thick=3
       oplot,[256+xyhex[mm,0]*hxp*.98,256+xyhex[mm,0]*hxp*1.01],[256+xyhex[mm,1]*hxp*.98,256+xyhex[mm,1]*hxp*1.01],color=color_vect(mm),thick=5
    endfor
stop

goto,startover
endall:

end

