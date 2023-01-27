const exifr = require('exifr')
const fs = require('fs')
const escapeHtml = require('escape-html')
const yaml = require('js-yaml');
const GM = require('./gm')
class Parser {
  constructor() {
    this.gm = new GM()
  }
  async parse(buf) {
    let parsed = await exifr.parse(buf, true)
    let attrs = {}
    if (parsed.ImageWidth) attrs.width = parsed.ImageWidth
    if (parsed.ImageHeight) attrs.height = parsed.ImageHeight

    let app
    if (parsed["sd-metadata"]) {
      app = "invokeai"
    } else if (parsed.parameters) {
      app = "automatic1111"
    }
    let meta = this.getMeta(parsed, attrs)
    for(let key in meta) {
      if (typeof meta[key] === "string") {
        meta[key] = escapeHtml(meta[key])
      }
    }

    return { ...attrs, ...meta, app }
  }

  async parseParametersText(parsed, parametersText) {

    const metadata = this.getMeta({
      ...parsed,
      parameters: parametersText
    })

    if (!metadata) {
      return parsed;
    }

    return { ...metadata, } // do not include "app" attribute because it's not possible to tell the app just from the parsed text file and there may be many other apps
  }

  convert(e, options) {
  /*
  automatic111 {
    width: 512,
    height: 512,
    Steps: '20',
    Sampler: 'DDIM',
    'CFG scale': '7',
    Seed: '76682',
    Size: '512x512',
    'Model hash': '38b5677b',
    'Variation seed': '458188939',
    'Variation seed strength': '0.06',
    prompt: 'lake ness monster, procam style'
  }
  invokeAI {
    "width": 512,
    "height": 512,
    "prompt": "a ((dog)) running in a park",
    "steps": 50,
    "cfg_scale": 7.5,
    "threshold": 0,
    "perlin": 0,
    "seed": 3561693215,
    "seamless": false,
    "hires_fix": false,
    "type": "txt2img",
    "postprocessing": null,
    "sampler": "k_lms",
    "variations": [],
    "weight": 1,
    "model": "stable diffusion",
    "model_weights": "stable-diffusion-1.5",
    "model_hash": "cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516",
    "app_id": "invoke-ai/InvokeAI",
    "app_version": "v2.2.0"
  }
  diffusionbee {
    "prompt": "mafia santa, retro style",
    "seed": "15982",
    "img_w": "768",
    "img_h": "768",
    "key": "0.6032483614978881",
    "guidence_scale": "7.5",
    "dif_steps": "50",
    "model_version": "custom_retrodiffusion2_fp16",
    "negative_prompt": "blurry, disfigured"
  }
  */



    const x = {}

    if (options && options.agent) {
      x["xmp:agent"] = options.agent
    } else if (e.agent) {
      x["xmp:agent"] = e.agent
    } else if (e.Agent) {
      x["xmp:agent"] = e.Agent
    } else if (e.app) {
      if (e.app === "invokeai") {
        if (e.app_id && e.app_version) {
          x["xmp:agent"] = `${e.app_id}`// ${e.app_version}`
        }
      } else if (e.app === "automatic1111") {
        x["xmp:agent"] = "automatic1111"
      }
    }


    if (options && options.width) {
      x["xmp:width"] = parseInt(options.width)
    } else if (e.width) {
      x["xmp:width"] = parseInt(e.width)
    } else if (e.img_w) {
      x["xmp:width"] = parseInt(e.img_w)
    } else if (e.Width) {
      x["xmp:width"] = parseInt(e.Width)
    }
    if (options && options.height) {
      x["xmp:height"] = parseInt(options.height)
    } else if (e.height) {
      x["xmp:height"] = parseInt(e.height)
    } else if (e.img_h) {
      x["xmp:height"] = parseInt(e.img_h)
    } else if (e.Height) {
      x["xmp:height"] = parseInt(e.Height)
    }

    if (options && options.cfg) {
      x["xmp:cfg_scale"] = options.cfg
    } else if (e["CFG scale"]) {
      x["xmp:cfg_scale"] = parseFloat(e["CFG scale"])
    } else if (e.cfg_scale) {
      x["xmp:cfg_scale"] = parseFloat(e.cfg_scale)
    } else if (e["Guidance Scale"]) {
      x["xmp:cfg_scale"] = parseFloat(e["Guidance Scale"])
    }

    if (options && options.seed) {
      x["xmp:seed"] = options.seed
    } else if (e.Seed) {
      x["xmp:seed"] = parseInt(e.Seed)
    } else if (e.seed) {
      x["xmp:seed"] = parseInt(e.seed)
    }

    if (options && options.negative) {
      x["xmp:negative_prompt"] = options.negative
    } else if (e["Negative prompt"]) {
      x["xmp:negative_prompt"] = e["Negative prompt"]
    } else {
      // invokeai does negative prompts differently (included in the prompt), so need to parse the prompt to extract negative prompts
      if (e.prompt && Array.isArray(e.prompt) && e.prompt[0].prompt) {
        // test for invokeAI negative prompt syntax
        let negative_chunks = []
        let positive_chunks = []
        for(let chunk of e.prompt) {
          if (chunk.prompt && typeof chunk.prompt === "string") {
            let matches = [...chunk.prompt.matchAll(/\[([^\]]+)\]/g)].map((m) => {
              return m[1]
            })
            if (matches.length > 0) {
              negative_chunks = negative_chunks.concat(matches) 
            }
            let positive = chunk.prompt.replaceAll(/\[[^\]]+\]/g, "").trim()
            positive_chunks.push(positive)
          }
        }
        if (negative_chunks.length > 0) {
          x["xmp:negative_prompt"] = escapeHtml(negative_chunks.join(", "))
        }
        x["xmp:prompt"] = escapeHtml(positive_chunks.join(" "))
      }
    }

    if (options && options.sampler) {
      x["xmp:sampler"] = options.sampler
    } else if (e.Sampler) {
      x["xmp:sampler"] = e.Sampler
    } else if (e.sampler) {
      x["xmp:sampler"] = e.sampler
    }

    if (options && options.steps) {
      x["xmp:steps"] = options.steps
    } else if (e.Steps) {
      x["xmp:steps"] = e.Steps
    } else if (e.steps) {
      x["xmp:steps"] = e.steps
    }

    if (!x["xmp:prompt"]) {
      x["xmp:prompt"] = e.prompt
    }

    if (options && options.model) {
      x["xmp:model_name"] = options.model
    } else if (e.model_version) {
      x["xmp:model_name"] = e.model_version
    } else if (e.model_weights) {
      x["xmp:model_name"] = e.model_weights
    } else if (e.Model) {
      x["xmp:model_name"] = e.Model
    }

    if (options && options.model_hash) {
      x["xmp:model_hash"] = options.model_hash
    } else if (e.Model_hash) {
      x["xmp:model_hash"] = e.Model_hash
    } else if (e["Model hash"]) {
      x["xmp:model_hash"] = e["Model hash"]
    } else if (e["model hash"]) {
      x["xmp:model_hash"] = e["model hash"]
    } else if (e.model_hash) {
      x["xmp:model_hash"] = e.model_hash
    }

    if (options && options.model_url) {
      x["xmp:model_url"] = options.model_url
    } else if (e.Model_url) {
      x["xmp:model_url"] = e.Model_url
    } else if (e.model_url) {
      x["xmp:model_url"] = e.model_url
    }

    if (options && options.subject) {
      x["dc:subject"] = options.subject
    } else if (e.subject) {
      x["dc:subject"] = e.subject
    } else if (e.Subject) {
      x["dc:subject"] = e.Subject
    }

    if (options && options.caption) {
      x["xmp:caption"] = options.caption
    } else if (e.caption) {
      x["xmp:caption"] = e.caption
    } else if (e.Caption) {
      x["xmp:caption"] = e.Caption
    }
    
    let keys = [
      "xmp:prompt",
      "xmp:sampler",
      "xmp:steps",
      "xmp:cfg_scale",
      "xmp:seed",
      "xmp:negative_prompt",
      "xmp:model_name",
      "xmp:model_hash",
      "xmp:model_url",
      "xmp:agent",
      "xmp:width",
      "xmp:height",
      "dc:subject",
      "xmp:caption"
    ]

    let list = []
    for(let key of keys) {
      if (x[key]) {
        list.push({ key, val: x[key] })
      } else {
        list.push({ key })
      }
    }

    return list

  }
  applyType(item) {
    let integers = ["xmp:steps", "xmp:seed", "xmp:width", "xmp:height"]
    let floats = ["xmp:cfg_scale"]
    if (integers.includes(item.key)) {
      return parseInt(item.val)
    }

    if (floats.includes(item.key)) {
      return parseFloat(item.val)
    }

    return item.val
  }
//  flatten (app, converted, filepath, btime, mtime) {
//    return {
//      app,
//      model: converted.model,
//      sampler: converted.image.sampler,
//      prompt: converted.image.prompt[0].prompt,
//      weight: converted.image.prompt[0].weight,
//      steps: converted.image.steps,
//      cfg_scale: converted.image.cfg_scale,
//      height: (converted.image.height ? converted.image.height : 'NULL'),
//      width: (converted.image.width ? converted.image.width : 'NULL'),
//      seed: converted.image.seed,
//      negative_prompt: converted.image.negative_prompt,
//      mtime,
//      btime,
//      path: filepath
//    }
//  }
  async serialize(root_path, file_path) {
    let info = await this.gm.get(file_path)
    let o = {}
    for(let item of info.parsed) {
      if (typeof item.val !== "undefined") {
        if (item.key.startsWith("xmp:")) {
          let key = item.key.replace("xmp:", "").toLowerCase()
          o[key] = this.applyType(item)
        } else if (item.key.startsWith("dc:")) {
          let key = item.key.replace("dc:", "").toLowerCase()
          if (Array.isArray(item.val) && item.val.length > 0) {
            o[key] = item.val
          }
        } else {
          o[item.key] = item.val
        }
      }
    }
    let stat = await fs.promises.stat(file_path)
    let btime = new Date(stat.birthtime).getTime()
    let mtime = new Date(stat.mtime).getTime()
    return { ...o, root_path, file_path, mtime, btime }
  }
  async serializeJpeg(root_path, file_path) {
    let info = await exifr.parse(file_path)
    
    let o = {}
    if (info) {
      try {
        for(let key of Object.keys(info)) {
          if (typeof info[key] !== "undefined") {
            
              o[key] = info[key]
            
          }
        }
      }
      catch (e) {
        console.log(e)
      }
      
    }
    let stat = await fs.promises.stat(file_path)
    let btime = new Date(stat.birthtime).getTime()
    let mtime = new Date(stat.mtime).getTime()
    return { ...o, root_path, file_path, mtime, btime }
  }
  getMeta(parsed, attr) {
    if (parsed["sd-metadata"]) {
      let p = JSON.parse(parsed["sd-metadata"])
      let image = p.image
      delete p.image
      if (attr) {
        if (attr.width) image.width = parseInt(attr.width)
        if (attr.height) image.height = parseInt(attr.height)
      }
      return { ...image, ...p, }
    } else if (parsed.parameters) {
      try {
        /**********************************************
        *
        *   try the following format first:
        *
        *   prompt
        *   key: val
        *   key: val
        *   key: val
        **********************************************/
        let lines = parsed.parameters.split(/\r?\n/)
        const attrs = yaml.load(lines.slice(1).join("\n"))
        return { prompt: lines[0], ...attrs }
      } catch (e) {
        /*******************************************************************
        *
        *   If the parse fails, try the following format (automatic1111):
        *
        *   prompt
        *   Negative prompt: .... (optional)
        *   key: val, key: val, key: val
        *
        *******************************************************************/
        let lines = parsed.parameters.split(/\r?\n/).filter((x) => {
          return x && x.length > 0
        })

        // last line is the attributes
        // last - 1 line
        //    if it starts with "Negative prompt:", is the negative prompt
        //    otherwise, there is no negative prompt
        // The rest lines are treated as the prompt

        let re = /([^:]+):([^:]+)(,|$|\n)/g
        let kvs = lines[lines.length-1]
        let items = [...kvs.matchAll(re)].map((item) => {
          return {
            key: item[1].trim(),
            val: item[2].trim()
          }
        })

        let negative
        if (lines.length >= 3 && lines[lines.length-2].startsWith("Negative prompt:")) {
          negative = true
          items.push({
            key: "Negative prompt",
            val: lines[lines.length-2].replace(/Negative prompt:[ ]?/, "")
          })
        }

        let promptStr
        if (negative) {
          promptStr = lines.slice(0, -2)
        } else {
          promptStr = lines.slice(0, -1)
        }

        let attrs = {}
        for(let kv of items) {
          attrs[kv.key] = kv.val
        }

        attrs.prompt = promptStr
        if (attr) {
          if (attr.width) attrs.width = parseInt(attr.width)
          if (attr.height) attrs.height = parseInt(attr.height)
        }
        return attrs
      }
    }
  }
}
module.exports = Parser
