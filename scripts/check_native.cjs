const fs=require('fs');
const {Battle,Teams}=require('../vendor/pokemon-showdown/dist/sim');
const data=JSON.parse(fs.readFileSync(process.argv[2]));
const b=new Battle({formatid:'gen9championsvgc2026regmc',seed:[1,2,3,4]});
b.setPlayer('p1',{name:'p1',team:Teams.import(data.team)});
b.setPlayer('p2',{name:'p2',team:Teams.import(data.team)});
for(const [side,choice] of data.choices) {
 if(!b.choose('p'+(side+1),choice)) throw Error('Reference rejected '+choice);
}
fs.writeFileSync(process.argv[3],JSON.stringify({ended:b.ended,log:b.log}));
