const { app, shell, BrowserWindow, ipcMain, dialog, session, clipboard } = require('electron')
const { fdir } = require("fdir");
const contextMenu = require('electron-context-menu');
const path = require('path')
const xmlFormatter = require('xml-formatter');
const fs = require('fs')
const os = require('os')
const express = require('express')
const fastq = require('fastq')
const getport = require('getport')
const Diffusionbee = require('./crawler/diffusionbee')
const Standard = require('./crawler/standard')
const Updater = require('./updater/index')
const GM = require('./crawler/gm')
const gm = new GM()
const is_mac = process.platform.startsWith("darwin")
const { PythonShell } = require('python-shell');
const unhandled = require('electron-unhandled');

unhandled({
  logger: (e) => {
    console.error("UNHANDLED ERROR", e);
  },
  showDialog: false,

});
var captionShell
var captionsToUpdate = []

contextMenu({ showSaveImageAs: true });
var mainWindow;
var theme = "default";
const updater = new Updater()
const titleBarOverlay = (theme) => {
  if (is_mac) {
    return false
  } else {
    if (theme === "dark") {
      return {
        color: "#111",
        symbolColor: "white"
      }
    } else if (theme === "default") {
      return {
        color: "white",
        symbolColor: "black"
      }
    }
    return {
      color: "white",
      symbolColor: "black"
    }
  }
}
function createWindow(port) {
  mainWindow = new BrowserWindow({
    titleBarStyle: "hidden",
    titleBarOverlay: titleBarOverlay(),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    },
  })
  //  mainWindow.webContents.openDevTools()
  mainWindow.loadURL(`http://localhost:${port}`)
  mainWindow.maximize();


  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https:')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });
}
var need_update = null
var default_sync_mode = "default"    // only synchronize once at launch
var current_sorter_code = 0


const VERSION = app.getVersion()
console.log("VERSION", VERSION)

const releaseFeed = "https://github.com/cocktailpeanut/breadboard/releases.atom"
const releaseURL = "https://github.com/cocktailpeanut/breadboard/releases"
const updateCheck = async () => {
  let res = await updater.check(releaseFeed)
  console.log("Feed", res)
  if (res.feed && res.feed.entry) {

    let latest = (Array.isArray(res.feed.entry) ? res.feed.entry[0] : res.feed.entry)
    if (latest.title === VERSION) {
      console.log("UP TO DATE", latest.title, VERSION)
    } else {
      console.log("Need to update to", latest)
      need_update = {
        $url: releaseURL,
        latest
      }
    }
  }
}
app.whenReady().then(async () => {
  await updateCheck().catch((e) => {
    console.log("update check error", e)
  })

  //  session.defaultSession.clearStorageData()   // for testing freshly every time
  const port = await new Promise((resolve, reject) => {
    getport(function (e, p) {
      if (e) throw e
      resolve(p)
    })
  })

  const server = express()
  server.use(express.static(path.resolve(__dirname, 'public')))
  server.use("/docs", express.static(path.resolve(__dirname, 'docs')))
  server.set('view engine', 'ejs');
  server.set('views', path.resolve(__dirname, "views"))
  server.get("/", async (req, res) => {
    let sync_mode = (req.query.synchronize ? req.query.synchronize : default_sync_mode)
    let sync_folder = (req.query.sync_folder ? req.query.sync_folder : "")
    if (req.query && req.query.sorter_code) {
      current_sorter_code = req.query.sorter_code
    }
    res.render("index", {
      platform: process.platform,
      query: req.query,
      version: VERSION,
      sync_mode,
      sync_folder,
      need_update,
      current_sorter_code,
      theme
    })
    if (default_sync_mode) default_sync_mode = false   // disable sync after the first time at launch
  })
  server.get("/settings", (req, res) => {
    res.render("settings", {
      platform: process.platform,
      version: VERSION,
      query: req.query,
      theme
    })
  })
  server.get("/favorites", (req, res) => {
    res.render("favorites", {
      platform: process.platform,
      version: VERSION,
      theme
    })
  })
  server.get("/help", (req, res) => {
    let items = [{
      name: "discord",
      description: "ask questions and share feedback",
      icon: "fa-brands fa-discord",
      href: "https://discord.gg/6MJ6MQScnX"
    }, {
      name: "twitter",
      description: "stay updated on Twitter",
      icon: "fa-brands fa-twitter",
      href: "https://twitter.com/cocktailpeanut"
    }, {
      name: "github",
      description: "feature requests and bug report",
      icon: "fa-brands fa-github",
      href: "https://github.com/cocktailpeanut/breadboard/issues"
    }]
    res.render("help", {
      items,
      platform: process.platform,
      version: VERSION,
      theme
    })
  })
  server.get('/file', (req, res) => {
    res.sendFile(req.query.file)
  })
  server.listen(port, () => {
    console.log(`Breadboard listening on port ${port}`)
  })
  ipcMain.handle('sync', async (event, rpc) => {
    console.log("## sync from rpc", rpc)
    let filter
    const queue = fastq.promise(async (msg) => {
      mainWindow.webContents.send('msg', msg)
    }, 1)
    if (rpc.paths) {
      let diffusionbee;
      let standard;
      for (let i = 0; i < rpc.paths.length; i++) {
        let { file_path, root_path } = rpc.paths[i]
        let res;
        try {
          if (/diffusionbee/g.test(root_path)) {
            if (!diffusionbee) {
              diffusionbee = new Diffusionbee(root_path)
              await diffusionbee.init()
            }
            res = await diffusionbee.sync(file_path, rpc.force)
          } else {
            if (!standard) {
              standard = new Standard(root_path)
              await standard.init()
            }
            res = await standard.sync(file_path, rpc.force)
          }
        } catch (e) {
          console.log("E", e)
        }
        if (res) {
          await queue.push({
            app: root_path,
            total: rpc.paths.length,
            progress: i,
            meta: res
          })
        } else {
          await queue.push({
            app: root_path,
            total: rpc.paths.length,
            progress: i,
          })
        }
      }
    } else if (rpc.root_path) {
      let filenames = await new fdir()
        .glob("**/*.png", "**/*.jpg", "**/*.jpeg")
        .withBasePath()
        .crawl(rpc.root_path)
        .withPromise()

      if (filenames.length > 0) {
        let crawler;
        if (/diffusionbee/g.test(rpc.root_path)) {
          crawler = new Diffusionbee(rpc.root_path)
        } else {
          crawler = new Standard(rpc.root_path)
        }
        await crawler.init()
        for (let i = 0; i < filenames.length; i++) {
          let filename = filenames[i]
          let stat = await fs.promises.stat(filename)
          let btime = new Date(stat.birthtime).getTime()
          if (!rpc.checkpoint || btime > rpc.checkpoint) {
            //console.log("above checkpoint", btime, rpc.checkpoint, filename)
            let res = await crawler.sync(filename, rpc.force)
            if (res) {
              if (!res.btime) res.btime = res.mtime
              await queue.push({
                app: rpc.root_path,
                total: filenames.length,
                progress: i,
                meta: res
              })
              continue;
            }
          }
          await queue.push({
            app: rpc.root_path,
            total: filenames.length,
            progress: i,
          })
        }
      } else {
        await queue.push({
          app: rpc.root_path,
          total: 1,
          progress: 1,
        })
      }
    }
  })
  ipcMain.handle("theme", (event, _theme) => {
    theme = _theme
    if (mainWindow.setTitleBarOverlay) {
      mainWindow.setTitleBarOverlay(titleBarOverlay(theme))
    }
  })
  ipcMain.handle('del', async (event, filenames) => {
    for (filename of filenames) {
      console.log("deleting", filename)
      await fs.promises.rm(filename).catch((e) => {
        console.log("error", e)
      })
    }
  })
  ipcMain.on('ondragstart', (event, filePaths) => {
    event.sender.startDrag({
      files: filePaths,
      icon: filePaths[0],
    })
  })
  ipcMain.handle('select', async (event) => {
    let res = await dialog.showOpenDialog({ properties: ['openDirectory', 'showHiddenFiles'] })
    if (!res.canceled && res.filePaths && res.filePaths.length > 0) {
      return res.filePaths
    }
  })
  ipcMain.handle('defaults', async (event) => {
    let home = os.homedir()
    return [
      path.resolve(home, "invokeai", "outputs"),
      path.resolve(home, ".diffusionbee", "images"),
    ]
  })
  ipcMain.handle('copy', (event, text) => {
    clipboard.writeText(text)
  })
  ipcMain.handle('gm', async (event, rpc) => {
    if (rpc.cmd === "set" || rpc.cmd === "rm") {
      let res = await gm[rpc.cmd](...rpc.args)
      return res
    }
  })
  ipcMain.handle('open', async (event, file_path) => {
    await shell.showItemInFolder(file_path)
  })
  ipcMain.handle('xmp', async (event, file_path) => {
    let res = await gm.get(file_path)
    return xmlFormatter(res.chunk.data.replace("XML:com.adobe.xmp\x00\x00\x00\x00\x00", ""), {
      indentation: "  "
    })
  })
  ipcMain.handle('docs', async (event, file_path) => {
    let modal = new BrowserWindow({
      parent: mainWindow,
      //      modal: true
    })
    modal.loadURL(`http://localhost:${port}/docs/doc.html`)
  })
  ipcMain.handle('debug', (event) => {
    mainWindow.webContents.openDevTools()
  })
  const log = fastq.promise(async (msg) => {
    mainWindow.webContents.send('caption_log', msg)
  }, 1)
  const finished = fastq.promise(async () => {
    mainWindow.webContents.send('caption_finished')
  }, 1)
  let captionAborted = false
  ipcMain.handle('captionBatch', async (event, args) => {
    if (captionAborted)
      return
    if (typeof args === 'undefined')
      return
    if (args.length === 0) {
      captionShell.end(async function (err) {
        if (err) {
          await log.push(`Error: ${JSON.stringify(err)}`)
          throw err;
        };
        await finished.push()
        console.log('finished');
      });
      return
    }
    let data = args.map((d) => {
      return {
        path: d.file_path,
        cap: d.caption
      }
    })
    data.forEach(async (item) => {
      if (!captionShell.terminated && !captionAborted) {
        try {
          await captionShell.send(JSON.stringify(item))
        } catch (e) {
          console.log(`ERROR WRITING TO PYTHON: ${e}`)
        }
        //console.log(`SENT: ${JSON.stringify(item)}`);

      }
    })

  })
  ipcMain.handle('captionAbort', async (event) => {
    if (captionShell.terminated || captionAborted)
      return
    captionAborted = true
    captionShell.kill()
    // captionShell.end(async function (err) {
    //   if (err){
    //     await log.push(`Error: ${JSON.stringify(err)}`)
    //     throw err;
    //   };
    //   await finished.push()
    //   console.log('finished');
    // });
  })
  ipcMain.handle('caption', async (event, args) => {
    captionAborted = false
    let argv = []
    for (const [key, value] of Object.entries(args)) {
      if (value == false)
        continue
      if (typeof value === 'boolean') {
        if (value)
          argv.push(`--${key}`)
      }
      else if (typeof value === 'string') {
        argv.push(`--${key}=${value}`)
      }
      else {
        argv.push(`--${key}=${value}`)
      }
    }
    let options = {
      mode: 'text',
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: 'utils',
      args: argv
    };
    captionShell = new PythonShell('caption.py', options);
    captionShell.on('stderr', async function (message) {
      console.log(message);
      await log.push(`${message}`)

    })
    captionShell.on('message', async function (message) {
      // TODO: Add new captions and write to disk
      console.log(message);
      if (!captionAborted)
        await log.push(`${message}`)
    })
    captionShell.send("check")
    await log.push("LOADING PYTHON... (This can take several minutes)")





  })
  PythonShell.run("config.py", { scriptPath: "utils" }, (err, results) => {
    if (err) throw err;
    console.log('results: %j', results);
  })
  createWindow(port)
  //  synchronize()
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow(port)
  })

})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})
