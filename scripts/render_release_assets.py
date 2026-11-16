"""Original release schematics. Equations are rendered by LaTeX, never Unicode approximations.

No demonstrations, model weights, measured attention, or experiment traces are used.
Requires Pillow, PyMuPDF, TeX Live (pdflatex), and DejaVu Sans.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
W, H, SCALE = 1440, 820, 2
C = dict(bg='#09131E', panel='#101F2D', edge='#263C4C', ink='#E5EFF5',
         muted='#8FA9BA', mint='#63E6BE', blue='#73AEFF', coral='#FF9F89',
         gold='#F6D177', grid='#172C3D', white='#FFFFFF')
FONTS = {}
EQ = {}
FORMULAS = {
    'context': r'\mathcal{T}=\{(c_i,\varepsilon_i^*)\}_{i=1}^{N}',
    'pair': r'(c_i,\varepsilon_i^*)',
    'predict': r'\widehat{\varepsilon}=F(c\mid\mathcal{T})',
    'readout': r'c=R_\theta(u)',
    'init': r'x_0=h+\sigma\widehat{\varepsilon}',
    'action': r'\widehat a=G_\theta(u,\widehat\varepsilon)',
    'freeze': r'\Delta\theta=0,\quad\Delta\phi=0',
    'update': r'\mathcal{T}\leftarrow\mathcal{T}\cup\{(c_{\rm new},\varepsilon_{\rm new}^*)\}',
    'forward': r'x_{k+1}=x_k+\Delta t_k\,v_\theta(x_k,u,t_k)',
    'inverse': r'x_k^{(j+1)}=x_{k+1}-\Delta t_k\,v_\theta(x_k^{(j)},u,t_k)',
    'noise': r'\varepsilon^*=(x_0^*-h)/\sigma',
    'residual': r'r=\operatorname{MSE}\!\left(G_\theta(u,\varepsilon^*),a^*\right)',
    'gate': r'r\leq\tau\;\Longrightarrow\;(c,\varepsilon^*)\in\mathcal{T}',
    'terminal': r'x_K=a^*',
    'scm': r'U\sim P_U,\qquad H_j=f_j(H_{\mathrm{pa}(j)})',
    'scm_x1': r'X_1=2U+1',
    'scm_x2': r'X_2=U+2',
    'scm_y': r'y=X_1-X_2',
    'columns': r'(X,y)=\operatorname{SelectColumns}(H)',
    'prior': r'\mathcal{D}\sim p_{\rm synthetic}(\mathcal{D})',
    'pretrain': r'\phi^*=\arg\min_\phi\;\mathbb{E}_{\mathcal D}\!\left[\mathcal L\!\left(F_\phi(X_q\mid X_s,y_s),y_q\right)\right]',
}


def font(size, bold=False):
    key = size, bold
    if key not in FONTS:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        candidates = [Path('/usr/share/fonts/truetype/dejavu') / name]
        candidates += [Path('/Library/Fonts') / name, Path.home() / 'Library/Fonts' / name]
        try:
            import matplotlib
            candidates.append(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / name)
        except ImportError:
            pass
        match = next((x for x in candidates if x.is_file()), None)
        if match is None:
            raise RuntimeError('Install DejaVu Sans before rendering.')
        FONTS[key] = ImageFont.truetype(str(match), size * SCALE)
    return FONTS[key]


def render_tex():
    """Compile all formula plates in one real LaTeX document; crop transparent pages."""
    latex = shutil.which('pdflatex')
    if latex is None:
        candidate = Path('/Library/TeX/texbin/pdflatex')
        if candidate.is_file():
            latex = str(candidate)
    if latex is None:
        raise RuntimeError('TeX Live with pdflatex is required.')
    with tempfile.TemporaryDirectory(prefix='restart-release-tex-') as tmp:
        folder = Path(tmp)
        tex = (r'\documentclass{article}\usepackage[paperwidth=36cm,paperheight=8cm,margin=0.3cm]{geometry}'
               r'\usepackage{amsmath,amssymb,xcolor}\pagestyle{empty}\definecolor{ink}{HTML}{E5EFF5}'
               r'\begin{document}\color{ink}\fontsize{22}{28}\selectfont' + '\n')
        tex += '\n\\newpage\n'.join(r'\null\vfill\[' + value + r'\]\vfill\null' for value in FORMULAS.values())
        tex += '\n' + r'\end{document}'
        (folder / 'equations.tex').write_text(tex)
        result = subprocess.run([latex, '-interaction=nonstopmode', '-halt-on-error', 'equations.tex'],
                                cwd=folder, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:])
        doc = fitz.open(folder / 'equations.pdf')
        if len(doc) != len(FORMULAS):
            raise RuntimeError('LaTeX formula pagination differs from the declared plates.')
        for name, page in zip(FORMULAS, doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=True)
            im = Image.frombytes('RGBA', (pix.width, pix.height), pix.samples)
            EQ[name] = im.crop(im.getbbox())
        doc.close()
    dest = OUT / 'equations'
    dest.mkdir(parents=True, exist_ok=True)
    for name, im in EQ.items():
        plate = Image.new('RGB', (im.width + 40, im.height + 32), C['bg'])
        plate.paste(im, (20, 16), im)
        plate.save(dest / (name + '.png'), optimize=True)
    (dest / 'equations.tex').write_text(tex)


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W*SCALE, H*SCALE), C['bg'])
        self.d = ImageDraw.Draw(self.im)
        for x in range(0, W, 36):
            self.line([(x, 0), (x, H)], 'grid')
        for y in range(0, H, 36):
            self.line([(0, y), (W, y)], 'grid')
    def text(self, xy, value, size=18, color='ink', bold=False, anchor=None):
        self.d.text(tuple(round(v*SCALE) for v in xy), value, font=font(size,bold), fill=C[color], anchor=anchor)
    def line(self, pts, color='edge', width=1):
        self.d.line([(round(x*SCALE),round(y*SCALE)) for x,y in pts],fill=C[color],width=width*SCALE,joint='curve')
    def box(self, xy, fill='panel', outline='edge', radius=12):
        self.d.rounded_rectangle(tuple(round(v*SCALE) for v in xy),radius=radius*SCALE,fill=C[fill],outline=C[outline],width=SCALE)
    def dot(self, xy, r=5, color='mint'):
        x,y=xy
        self.d.ellipse(tuple(round(v*SCALE) for v in (x-r,y-r,x+r,y+r)),fill=C[color])
    def arrow(self, a, b, color='mint', width=2):
        self.line([a,b],color,width)
        angle=math.atan2(b[1]-a[1],b[0]-a[0])
        pts=[(b[0]-10*math.cos(angle)+s*5*math.sin(angle), b[1]-10*math.sin(angle)-s*5*math.cos(angle)) for s in (-1,1)]
        self.line([pts[0],b,pts[1]],color,width)
    def eq(self, name, xy, width=400, height=40, color='ink'):
        im=EQ[name].copy()
        alpha=im.getchannel('A')
        plate=Image.new('RGBA',im.size,C[color]); plate.putalpha(alpha)
        factor=min(width*SCALE/im.width,height*SCALE/im.height)
        plate=plate.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.LANCZOS)
        self.im.paste(plate,(round(xy[0]*SCALE),round(xy[1]*SCALE)),plate)
    def header(self, chapter, title, subtitle):
        self.text((42,25),'RESTART  /  '+chapter,15,'mint',True)
        self.text((42,61),title,36,bold=True)
        self.text((42,113),subtitle,18,'muted')
        self.text((1395,28),'METHOD SCHEMATIC',13,'muted',anchor='rt')
        self.line([(42,153),(1398,153)])
    def footer(self, names, p, note):
        active=min(int(p*len(names)),len(names)-1)
        for i,label in enumerate(names):
            x=42+i*(1356/len(names)); right=x+1356/len(names)-18
            self.line([(x,744),(right,744)],'mint' if i==active else 'edge',3)
            self.text((x,758),f'{i+1:02d}  {label}',15,'mint' if i==active else 'muted',i==active)
        self.text((42,799),note,12,'muted')
    def finish(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS)


def move(c, path, phase, color='mint'):
    phase=max(0,min(1,phase))
    lengths=[math.dist(a,b) for a,b in zip(path,path[1:])]
    distance=phase*sum(lengths)
    for i,length in enumerate(lengths):
        if distance<=length or i==len(lengths)-1:
            a,b=path[i],path[i+1];t=distance/length if length else 0
            pos=(a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]))
            c.dot(pos,8,color);c.dot(pos,3,'white');break
        distance-=length



def matrix(c,x,y,rows,cols,cell=10,phase=0,active=False):
    for i in range(rows):
        for j in range(cols):
            v=math.sin(i*.83+j*.52+phase*5)
            color='mint' if v>.25 else 'blue' if v<-.25 else 'edge'
            if active and i==rows-1: color='gold'
            c.box((x+j*cell,y+i*cell,x+j*cell+cell-3,y+i*cell+cell-3),color,color,2)


def ease(t):
    t = max(0., min(1., t))
    return t * t * (3 - 2 * t)


def dash(c, points, color='muted', width=2):
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        n = max(1, int(length / 8))
        for i in range(0, n, 2):
            c.line([(a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n),
                    (a[0]+(b[0]-a[0])*min(i+1,n)/n,
                     a[1]+(b[1]-a[1])*min(i+1,n)/n)], color, width)


def ring(c, xy, radius=18, color='gold', width=2):
    x,y=xy
    c.d.ellipse(tuple(round(v*SCALE) for v in (x-radius,y-radius,x+radius,y+radius)),
                outline=C[color], width=width*SCALE)


def lock(c,x,y,color='mint'):
    c.d.arc(tuple(round(v*SCALE) for v in (x+5,y,x+23,y+25)),180,360,
            fill=C[color],width=3*SCALE)
    c.box((x,y+12,x+28,y+35),'panel',color,4)
    c.dot((x+14,y+23),2,color)


def gripper(c,x,y,color='ink',small=False):
    k=.55 if small else 1.
    c.line([(x,y-28*k),(x,y-7*k)],color,3)
    c.line([(x-14*k,y+8*k),(x-14*k,y-7*k),(x+14*k,y-7*k),(x+14*k,y+8*k)],color,3)


def scene(c, bounds, *, progress=0., target_color='gold', caption=None, ghost=False, variant=0):
    """A flat, explicitly schematic workspace; not robot evaluation footage."""
    x,y,w,h=bounds
    c.box((x,y,x+w,y+h),'bg','edge',10)
    start=(x+w*(.16+.045*(variant%3)),y+h*(.61-.07*(variant%2)))
    target=(x+w*(.77-.055*(variant%3)),y+h*(.40+.075*(variant%3)))
    ring(c,target,19,target_color,2)
    c.line([(target[0]-24,target[1]),(target[0]+24,target[1])],target_color)
    c.line([(target[0],target[1]-24),(target[0],target[1]+24)],target_color)
    def route(t):return (start[0]+(target[0]-start[0])*t,start[1]+(target[1]-start[1])*t-27*math.sin(math.pi*t))
    path=[route(t/40) for t in range(41)]
    c.line(path,'edge',2)
    if ghost:
        miss=(target[0]-13,target[1]+47)
        dash(c,[start,(start[0]+w*.3,start[1]-30),miss],'coral',2)
        ring(c,miss,9,'coral')
        c.text((miss[0]+18,miss[1]-7),'default',12,'coral')
    t=ease(progress)
    reached=[route(i/40) for i in range(max(2,int(t*40)+1))]
    if t>0:c.line(reached,'mint',3)
    pt=route(t)
    c.box((pt[0]-9,pt[1]-9,pt[0]+9,pt[1]+9),'mint','mint',3)
    gripper(c,pt[0],pt[1]-14,small=w<250)
    if caption:c.text((x+14,y+13),caption,12,'muted',True)
    return start,target


def caption(c,number,title,detail,color='mint'):
    c.box((42,631,1398,719),'panel',color,12)
    c.text((65,650),number,28,color,True)
    c.text((124,646),title,24,'ink',True)
    c.text((125,685),detail,16,'muted')


def context_frame(p):
    c=Canvas(); t=p*32
    c.header('CONTEXT-CONDITIONED SAMPLING','Learn the start. Keep the policy.',
             'Expert demonstrations build support context. Current observations and history define the new query.')
    colors=['gold','coral','gold']
    def route(q,k): return (167+1080*q,467+k*43+39*math.sin(math.pi*q)*(1-.13*k))
    for k in range(-3,4):
        c.line([route(i/100,k) for i in range(101)],'edge',2)
        for q in (.25,.52,.79):c.arrow(route(q-.015,k),route(q+.015,k),'edge',2)
    c.text((165,305),'SAMPLING START',13,'muted',True)
    c.text((1248,305),'EXPERT ACTIONS',13,'gold',True,anchor='rt')
    c.text((716,563),'FROZEN POLICY / SAME SAMPLING MAP',15,'muted',True,'mt')
    lock(c,679,592);c.eq('freeze',(728,596),224,25,color='muted')
    c.text((54,181),'SUPPORT CONTEXT',14,'gold',True)
    c.text((54,206),'Expert-derived paired examples',12,'muted')
    c.text((54,230),'Stored across demonstrations',12,'muted')
    for i,col in enumerate(colors):
        x=310+i*172; visible=t>=10
        c.box((x,178,x+153,262),'panel',col if visible else 'edge',9)
        if visible:
            for j in range(5):
                h=11+((i*13+j*7)%24)
                c.line([(x+17+j*10,235),(x+17+j*10,235-h)],col,3)
            c.line([(x+77,192),(x+77,248)],'edge')
            c.line([(x+91,243),(x+138,243)],'edge')
            c.line([(x+91,243),(x+91,193)],'edge')
            c.dot((x+115,205+i*14),5,col)
        else:c.text((x+76,214),'pending example',11,'muted',anchor='mt')
    c.eq('context',(850,196),290,30,color='gold' if t>=10 else 'muted')
    c.text((850,241),'SUPPORT LOADED' if t>=16 else 'SEND SUPPORT' if t>=12 else 'BUILD OFFLINE',12,
           'gold' if t>=12 else 'muted',True)
    # History is native current-call information, visually separate from support.
    c.box((54,641,349,719),'panel','blue',9)
    c.text((72,650),'CURRENT QUERY',12,'blue',True)
    c.text((72,672),'Observation + instruction + history',12,'ink')
    c.eq('readout',(74,694),230,19,color='blue')
    c.box((473,641,831,719),'panel','mint' if t>=18 else 'edge',9)
    if t<18:
        c.text((652,655),'CONTEXT PREDICTOR',15,'muted',True,'mt')
        c.text((652,690),'Waiting for support + current query',12,'muted',anchor='mt')
    else:
        c.eq('predict',(503,653),300,28,color='mint')
        c.text((652,695),'PREDICTING' if t<20 else 'PREDICTION READY',12,'mint',True,'mt')
    c.box((960,641,1398,719),'panel','blue',9)
    c.text((978,650),'NATIVE HISTORY ANCHOR h',12,'blue',True)
    c.eq('init',(978,680),392,27,color='mint' if t>=23 else 'muted')
    # Support travels once along a visible bus and fully arrives before query/compute.
    bus=[(653,277),(1420,277),(1420,732),(653,732),(653,720)]
    if t>=12:
        for i in range(3):c.line([(386+i*172,263),(386+i*172,277),(653,277)],'gold',2)
        c.line(bus,'gold' if t<16 else 'edge',2)
        if t<16:move(c,bus,(t-12)/4,'gold')
        else:c.dot((653,720),4,'gold')
    c.arrow((349,679),(469,679),'blue' if t>=16 else 'edge',2)
    if 16<=t<18:move(c,[(349,679),(469,679)],(t-16)/2,'blue')
    chosen=route(0,.48)
    output=[(473,702),(402,702),(402,742),(28,742),(28,chosen[1]),chosen]
    if t>=20:
        c.line(output,'mint',2)
        if t<23:move(c,output,(t-20)/3,'mint')
    # Expert inversion and context construction use the same persistent flow map.
    for i,col in enumerate(colors):
        k=i-1; path=[route(j/100,k) for j in range(101)]
        ring(c,path[-1],11,col)
        c.text((1266,path[-1][1]-7),'expert '+str(i+1),12,col)
        if 2<=t<7:
            f=(t-2)/5
            c.line([route(j/100,k) for j in range(int((1-f)*100),101)],col,3)
            move(c,list(reversed(path)),f,col)
        elif t>=7:
            ring(c,path[0],8,col)
            if t<10:
                dest=(386+i*172,262)
                transfer=[path[0],(path[0][0],286),(dest[0],286),dest]
                c.line(transfer,'edge');move(c,transfer,(t-7)/3,col)
    if t>=23:
        ring(c,chosen,15,'mint');c.dot(chosen,5,'mint')
        c.text((165,chosen[1]+25),'history + predicted noise',13,'mint')
    if t>=25:
        path=[route(j/100,.48) for j in range(101)]
        f=min(1,(t-25)/5)
        c.line([route(j/100,.48) for j in range(max(2,int(f*100)+1))],'mint',4)
        move(c,path,f,'mint')
        if t>=30:
            ring(c,path[-1],15,'mint')
            c.text((1248,path[-1][1]+27),'generated action',14,'mint',anchor='rt')
    if t<2:title='01  Expert demonstrations provide the target actions.'
    elif t<7:title='02  Reverse the frozen sampler to recover their starts.'
    elif t<12:title='03  Store situation–start pairs as support context.'
    elif t<16:title='04  Deliver the complete support context to the predictor.'
    elif t<18:title='05  Deliver the current observation, instruction and history.'
    elif t<20:title='06  Both inputs are ready. Now predict the noise.'
    elif t<23:title='07  Send the predicted noise to the sampling input.'
    elif t<25:title='08  Combine it with the native history anchor.'
    elif t<30:title='09  Run the same frozen policy from this start.'
    else:title='10  The new action is ready.'
    c.text((54,759),title,23,'ink',True)
    c.text((54,798),'Gold/coral: expert support. Blue: current query and history. Mint: prediction and generated action. Schematic sampling space.',12,'muted')
    return c.finish()


def inversion_frame(p):
    c=Canvas();t=p*32
    c.header('EXPERT INVERSION / BUILDING SUPPORT','Start with the expert action. Recover its sampling start.',
             'Reverse locally, replay forward, then store the situation together with its recovered start.')
    c.text((54,183),'RECORDED SITUATION',13,'blue',True)
    c.box((54,214,365,282),'panel','blue')
    c.text((72,226),'Observation + instruction + history',13,'ink')
    c.eq('readout',(72,253),265,22,color='blue')
    c.box((1005,214,1398,282),'panel','gold' if t>=25 else 'edge')
    c.text((1024,225),'SUPPORT CONTEXT',13,'gold' if t>=25 else 'muted',True)
    if t>=28:c.eq('pair',(1024,252),320,23,color='gold')
    else:c.text((1024,253),'Receiving checked pair' if t>=25 else 'Wait for inverse + replay check',14,'muted')
    def route(q,k=0):return (166+1082*q,440+k*42-38*math.sin(math.pi*q))
    for k in (-2,-1,0,1,2):
        c.line([route(i/100,k) for i in range(101)],'edge',2)
        for q in (.25,.5,.75):c.arrow(route(q-.015,k),route(q+.015,k),'edge',2)
    c.text((166,313),'RECOVERED START',13,'muted',True)
    c.text((1248,313),'EXPERT ACTION',13,'gold',True,anchor='rt')
    ring(c,route(1),14,'gold');c.dot(route(1),5,'gold')
    pts=[route(i/4) for i in range(5)]
    for pt in pts:c.dot(pt,4,'muted')
    if 2<=t<14:
        q=(t-2)/3;step=min(int(q),3);f=q-step
        start,end=pts[4-step],pts[3-step]
        for j in range(step):c.line([pts[4-j],pts[3-j]],'gold',3)
        if f<.6:
            c.arrow(start,end,'gold',3);move(c,[start,end],f/.6,'gold')
        else:
            c.line([start,end],'gold',3)
            jj=min(int((f-.6)/.4*4),3)
            previous=None
            for j in range(jj+1):
                off=36*(-.45)**j;v=(end[0]+abs(off),end[1]+off)
                if previous:c.line([previous,v],'gold',2)
                ring(c,v,4,'gold');previous=v
    if t>=14:
        ring(c,route(0),13,'gold')
        if t>=17:
            f=min(1,(t-17)/6)
            c.line([route(i/100) for i in range(max(2,int(f*100)+1))],'mint',3)
            move(c,[route(i/100) for i in range(101)],f,'mint')
    c.text((716,558),'FROZEN POLICY / SAME SAMPLING MAP',15,'muted',True,'mt')
    lock(c,679,588);c.eq('freeze',(725,592),235,25,color='muted')
    c.box((54,645,915,721),'panel','gold' if t<17 else 'mint')
    if t<17:c.eq('inverse',(76,665),815,35,color='gold')
    else:c.eq('noise',(76,665),815,35,color='mint')
    if t>=25:
        # Two separate inputs meet at the stored pair; both routes use gutters.
        a=[(365,248),(943,248),(943,164),(1199,164),(1199,214)]
        b=[route(0),(28,440),(28,295),(985,295),(985,266),(1005,266)]
        c.line(a,'blue',2);c.line(b,'gold',2)
        if t<28:move(c,a,(t-25)/3,'blue');move(c,b,(t-25)/3,'gold')
    if t<2:title='01  Fix the recorded situation and expert action.'
    elif t<14:title='02  Reverse each step, with local fixed-point corrections.'
    elif t<17:title='03  The recovered start is ready for verification.'
    elif t<23:title='04  Replay forward through the same frozen policy.'
    elif t<25:title='05  Forward replay verifies the recovered start.'
    elif t<28:title='06  Pair the recorded situation with its recovered start.'
    else:title='07  A new expert-derived support example is ready.'
    c.text((54,758),title,23,'ink',True)
    c.text((54,798),'Gold: expert supervision. Blue: recorded condition. Mint: forward replay. Schematic projection; no measured convergence claim.',12,'muted')
    return c.finish()


def scm_frame(p):
    c=Canvas();t=p*32;c.header('THE SYNTHETIC PRIOR','Invent a world. Hide an answer. Learn to infer it.',
        'Before robot adaptation, synthetic causal mechanisms generate supervised tasks for the context predictor.')
    for bounds in ((42,180,460,607),(492,180,928,607),(960,180,1398,607)):c.box(bounds)
    c.text((64,202),'1  GENERATE A TINY WORLD',16,'blue',True)
    # Actual values of a deliberately simple illustrative SCM, not model outputs.
    index=min(int(t/3),3) if t<12 else 4
    revealed=t>=24
    loaded=t>=18
    predicted=t>=22
    u=index; x1=2*u+1; x2=u+2; y=x1-x2
    positions=[(249,279),(143,388),(354,388),(249,507)]
    edges=[(0,1),(0,2),(1,3),(2,3)]
    for i,(a,b) in enumerate(edges):
        start=(positions[a][0],positions[a][1]+25)
        end=(positions[b][0],positions[b][1]-25)
        c.arrow(start,end,'edge',2)
        if t<12:move(c,[start,end],max(0,min(1,((t%3)-i*.3)/1.6)),'blue')
    for i,(x,yy) in enumerate(positions):
        c.box((x-51,yy-25,x+51,yy+25),'bg','blue' if i<3 else 'gold',10)
        values = ['U',str(x1),str(x2),'?' if index == 4 and not revealed else str(y)]
        c.text((x,yy-12),values[i] if i else f'U = {u}',22,'blue' if i<3 else 'gold',True,'mt')
    c.eq('scm_x1',(65,230),146,21,color='blue')
    c.eq('scm_x2',(287,230),146,21,color='blue')
    c.eq('scm_y',(180,548),165,23,color='gold')
    c.text((65,582),'One illustrative mechanism; not a robot model.',12,'muted')
    c.arrow((460,388),(490,388),'gold' if t<12 else 'edge',2)
    c.text((514,202),'2  TURN IT INTO EXAMPLES',16,'gold',True)
    c.text((541,258),'X1',20,'blue',True);c.text((659,258),'X2',20,'blue',True);c.text((806,258),'y',20,'gold',True)
    c.line([(516,299),(902,299)])
    visible=min(int(t/3),4)
    for i in range(4):
        yy=322+i*45
        if i<visible:
            if i==visible-1 and t<12:c.box((515,yy-6,903,yy+32),'bg','blue',5)
            c.text((554,yy),str(2*i+1),21,'ink',anchor='mt')
            c.text((672,yy),str(i+2),21,'ink',anchor='mt')
            c.text((814,yy),str(i-1),21,'gold',anchor='mt')
    c.box((515,508,904,568),'bg','blue',8)
    c.text((532,520),'ANSWER' if revealed else 'QUERY',11,'gold' if revealed else 'blue',True)
    c.text((602,529),'9',22,'blue',True,'mt');c.text((704,529),'6',22,'blue',True,'mt')
    c.text((814,529),'3' if revealed else '?',25,'gold',True,'mt')
    c.text((515,584),'Target revealed only for training supervision.' if revealed else
           'The query target is held out from the predictor.',12,'muted')
    # Support and query enter distinct ports and finish before inference.
    support=[(928,345),(944,345),(944,286),(990,286)]
    query=[(904,538),(944,538),(944,335),(990,335)]
    if t>=12:c.line(support,'gold' if t<16 else 'edge',2)
    if 12<=t<16:move(c,support,(t-12)/4,'gold')
    if t>=16:c.line(query,'blue' if t<18 else 'edge',2)
    if 16<=t<18:move(c,query,(t-16)/2,'blue')
    c.text((982,202),'3  LEARN TO FILL THE GAP',16,'mint',True)
    c.box((990,262,1369,359),'bg','mint',12)
    c.text((1178,283),'CONTEXT PREDICTOR',19,'mint',True,'mt')
    c.text((1178,324),'waiting for both inputs' if not loaded else 'predicting from context' if not predicted else 'prediction ready',15,'ink',anchor='mt')
    if predicted:
        c.arrow((1178,359),(1178,413),'mint')
        if t<24:move(c,[(1178,359),(1178,413)],(t-22)/2,'mint')
    c.box((995,414,1365,480),'bg','gold' if revealed else 'edge',9)
    c.text((1012,428),'Target revealed for supervision.' if revealed else 'Target remains hidden.',16,'gold' if revealed else 'muted',True)
    c.text((1012,455),'Compare prediction with the true target.' if revealed else 'Wait for the completed prediction.',14,'muted')
    if t>=26:
        feedback=[(1365,450),(1384,450),(1384,240),(1178,240),(1178,262)]
        c.line(feedback,'gold',2)
        if t<29:move(c,feedback,(t-26)/3,'gold')
    c.text((982,518),'Repeat across many sampled worlds.',16,'ink',True)
    c.text((982,550),'Freeze after pretraining.' if t>=29 else 'Pretraining can update weights.',16,'mint',True)
    c.text((982,582),'Pretraining updates weights; robot adaptation does not.',12,'muted')
    if t<12:title='01  Generate support examples from a sampled mechanism.'
    elif t<16:title='02  Deliver support rows, including their known labels.'
    elif t<18:title='03  Send the query features. Keep its target hidden.'
    elif t<22:title='04  Both inputs are ready. Predict the held-out target.'
    elif t<24:title='05  Deliver the prediction before revealing the answer.'
    elif t<26:title='06  Reveal the true target and compute supervision.'
    elif t<29:title='07  Use supervision to train across synthetic mechanisms.'
    else:title='08  Reuse the pretrained predictor with frozen weights.'
    c.text((54,677),title,22,'ink',True)
    c.text((54,718),'Gold: known supervision. Blue: query features. Mint: prediction. Feedback is used during pretraining only.',13,'muted')
    c.text((54,798),'Toy SCM values follow the displayed rule. No measured predictor output is shown. Robot adaptation uses frozen weights.',12,'muted')
    return c.finish()


def save_animation(name,renderer,count=None):
    count = count or 160
    frames=[renderer(i/(count-1)) for i in range(count)]
    sample=Image.new('RGB',(W,H*3))
    for i,j in enumerate((0,count//2,count-1)):sample.paste(frames[j],(0,i*H))
    palette=sample.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
    indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    indexed[0].save(OUT/(name+'.gif'),save_all=True,append_images=indexed[1:],duration=200,loop=0,optimize=True,disposal=1)
    frames[int(count*.96)].save(OUT/(name+'.png'),optimize=True)
    print(f'{name}: {(OUT/(name+".gif")).stat().st_size:,} bytes',flush=True)
    return frames


def main():
    OUT.mkdir(parents=True,exist_ok=True);render_tex()
    for name, renderer in (('restart-context', context_frame),
                           ('restart-inversion', inversion_frame),
                           ('restart-scm', scm_frame)):
        save_animation(name, renderer)


if __name__ == '__main__':
    main()
