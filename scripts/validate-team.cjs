const fs = require('fs');
const {Teams, TeamValidator, Dex} = require('../vendor/pokemon-showdown/dist/sim');
const config = JSON.parse(fs.readFileSync(process.argv[2] || 'configs/local.json'));
const format = Dex.formats.get(config.format);
if (!format.exists || format.gameType !== 'doubles') throw Error('Unsupported doubles format');
const team = Teams.import(fs.readFileSync(config.team, 'utf8'));
const errors = new TeamValidator(config.format).validateTeam(team);
if (errors) { console.error(errors); process.exit(1); }
console.log(JSON.stringify({format: format.id, mod: format.mod, team: team.map(p=>p.species), valid: true}));
