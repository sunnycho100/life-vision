"""Browser-only smoke checks for calibrated pool surface coordinates."""
import argparse
import asyncio
from playwright.async_api import async_playwright


async def main(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-swiftshader"])
        page = await browser.new_page()
        await page.goto(url)
        result = await page.evaluate("""async () => {
          const {surfacePosition,homography,project} = await import('./core.mjs');
          const assert = (value, message) => {if (!value) throw Error(message);};
          const close = (a,b) => Math.abs(a-b)<1e-6;
          // Oblique quadrilateral is the image of a 20 by 30 m rectangle.
          const corners = [[.1,.1],[.9,.2],[.8,.9],[.2,.8]], config = {corners,width:20,length:30,water:1.2,measured:true};
          const boxAt = p => [p[0]-.005,p[1]-.01,p[0]+.005,p[1]];
          const inverse=homography([[0,0],[20,0],[20,30],[0,30]],corners);
          const checks = [[[.1,.1],[0,0]],[[.9,.2],[20,0]],[[.8,.9],[20,30]],[[.2,.8],[0,30]],[project(inverse,10,15),[10,15]]];
          for (const [image, expected] of checks) {
            const position=surfacePosition({bbox_xyxy_normalized:boxAt(image)},config);
            assert(position && close(position.x,expected[0]) && position.y===1.2 && close(position.z,expected[1]),'corner/center mapping');
            assert(position.y===1.2 && close(position.z,expected[1]) && position.height_assumed && position.calibration==='user-measured','surface metadata');
          }
          assert(surfacePosition({bbox_xyxy_normalized:boxAt([.04,.5])},config)===null,'outside pool');
          for (const bad of [{...config,width:0},{...config,length:Infinity},{...config,water:NaN},{...config,water:-1},
            {...config,corners:[[0,0],[1,1],[0,1],[1,0]]},{...config,corners:undefined},{...config,corners:[[0,0],[1],[0,1],[1,0]]}])
            assert(surfacePosition({bbox_xyxy_normalized:boxAt([.5,.5])},bad)===null,'invalid calibration');
          const approximate=surfacePosition({bbox_xyxy_normalized:boxAt([.5,.5])},{...config,measured:false});
          assert(approximate.calibration==='approximate' && approximate.y===1.2 && [approximate.x,approximate.y,approximate.z].every(Number.isFinite),'finite approximate result');
          const {PoolTwin} = await import('./twin.js');
          const canvas = document.createElement('canvas'); canvas.style.cssText='width:640px;height:480px;position:fixed;left:0;top:0;z-index:10000'; document.body.append(canvas);
          let selected=null; const twin = new PoolTwin(canvas,document.querySelector('video'),p=>selected=p);
          twin.configure({...config,depth:1.5}); twin.resizeCanvas();
          const detections=[{bbox_xyxy_normalized:boxAt(project(inverse,10,15)),confidence:.6,class_name:'person'}];
          twin.updatePeople(detections); twin.render(performance.now());
          assert(canvas.dataset.markerCount==='1' && twin.people.children.length===3,'symbolic 3D marker');
          const pin=twin.people.children[1];
          assert(close(pin.position.x,0) && close(pin.position.z,0) && close(pin.position.y,1.7),'Three.js surface axes');
          twin.updatePeople(detections); assert(twin.people.children[1]===pin,'unchanged markers reused');
          const screen=pin.position.clone().project(twin.camera), rect=canvas.getBoundingClientRect();
          const event={clientX:rect.left+(screen.x+1)*rect.width/2,clientY:rect.top+(1-screen.y)*rect.height/2,pointerId:1,bubbles:true};
          // Actual browser pointer events below exercise raycasting and the selection callback.
          window.surfaceTest={twin,canvas,event,getSelected:()=>selected};
          return 'surface position corner, center, bounds, invalid calibration and 3D placement checks passed';
        }""")
        print(result, flush=True)
        point = await page.evaluate("({x:surfaceTest.event.clientX,y:surfaceTest.event.clientY})")
        await page.mouse.click(point["x"], point["y"])
        await page.evaluate("""() => {
          const p=surfaceTest.getSelected();
          if(!p || Math.abs(p.x-10)>1e-6 || Math.abs(p.z-15)>1e-6 || p.y!==1.2) throw Error('3D marker selection failed');
          surfaceTest.twin.updatePeople([]);
          if(surfaceTest.getSelected()!==null || surfaceTest.canvas.dataset.markerCount!=='0') throw Error('stale selection not cleared');
        }""")
        print("3D marker raycast selection and stale-observation clearing passed", flush=True)
        await browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    asyncio.run(main(parser.parse_args().url))
