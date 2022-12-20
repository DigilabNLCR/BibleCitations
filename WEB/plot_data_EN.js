// chart setup
const labels = [1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939]

const borderWidth = 2;
const tension = 0.2;

const dataForPlottingTop = 
[{
		verseId: 'Matt 5:38',
		citations: [2, 7, 5, 5, 8, 7, 7, 7, 13, 2, 5, 1, 4, 5, 2],
		backgroundColor: 'rgb(0, 83, 101, 0.6)',
		borderColor: 'rgb(0, 83, 101, 0.9)',
		hoverBackgroundColor: 'rgb(0, 83, 101, 1)',
		verseText: 'Slyšeli jste, že bylo řečeno: „Oko za oko, zub za zub.“'
	},{
		verseId: 'Matt 3:3/John 1:23/Mark 1:3/Luke 3:4/Isa 40:3',
		citations: [4, 9, 11, 6, 5, 6, 9, 1, 4, 0, 3, 1, 1, 0, 5],
		backgroundColor: 'rgb(0, 118, 79, 0.6)',
		borderColor: 'rgb(0, 118, 79, 0.9)',
		hoverBackgroundColor: 'rgb(0, 118, 79, 1)',
		verseText: 'jakož psáno jest v knize řečí proroka Isaiáše: „Hlas volajícího na poušti: Připravte cestu Páně, přímé čiňte stezky jeho;'
	},{
		verseId: 'Luke 2:14',
		citations: [7, 10, 11, 8, 4, 12, 8, 9, 11, 4, 2, 1, 5, 2, 3],
		backgroundColor: 'rgb(159, 101, 30, 0.6)',
		borderColor: 'rgb(159, 101, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 101, 30, 1)',
		verseText: 'Sláva na výsostech Bohu, a na zemi pokoj, lidem dobrá vůle.'
	},{
		verseId: 'Matt 11:28',
		citations: [1, 6, 4, 11, 2, 1, 4, 1, 2, 2, 1, 0, 2, 1, 2],
		backgroundColor: 'rgb(159, 36, 30, 0.6)',
		borderColor: 'rgb(159, 36, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 36, 30, 1)',
		verseText: '„Pojďte ke mně všichni, kteří těžce pracujete a jste přetíženi, a já vám dám odpočinek.'
	},{
		verseId: 'Matt 6:11/Luke 11:3',
		citations: [4, 3, 3, 0, 4, 6, 2, 3, 7, 3, 3, 1, 6, 5, 4],
		backgroundColor: 'rgb(0, 139, 168, 0.6)',
		borderColor: 'rgb(0, 139, 168, 0.9)',
		hoverBackgroundColor: 'rgb(0, 139, 168, 1)',
		verseText: 'Chléb náš vezdejší dej nám dnes.'
	},{
		verseId: 'Exod 20:15/Deut 5:19',
		citations: [6, 3, 7, 3, 5, 3, 6, 1, 1, 2, 2, 2, 2, 0, 4],
		backgroundColor: 'rgb(0, 168, 113, 0.6)',
		borderColor: 'rgb(0, 168, 113, 0.9)',
		hoverBackgroundColor: 'rgb(0, 168, 113, 1)',
		verseText: 'Nepokradeš.'
	},{
		verseId: 'Matt 16:18',
		citations: [16, 7, 4, 2, 3, 3, 6, 1, 2, 0, 1, 2, 2, 0, 0],
		backgroundColor: 'rgb(227, 144, 43, 0.6)',
		borderColor: 'rgb(227, 144, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 144, 43, 1)',
		verseText: 'I já pravím tobě: Ty jsi Petr (t. j. skála), a na té skále vzdělám cirkev svou, a brány pekelné jí nepřemohou.'
	},{
		verseId: 'Exod 20:13/Deut 5:17',
		citations: [11, 6, 11, 6, 16, 7, 6, 7, 5, 11, 6, 7, 2, 3, 3],
		backgroundColor: 'rgb(227, 52, 43, 0.6)',
		borderColor: 'rgb(227, 52, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 52, 43, 1)',
		verseText: 'Nezabiješ.'
	},{
		verseId: 'Luke 20:25',
		citations: [6, 4, 3, 5, 1, 4, 5, 0, 1, 2, 2, 1, 2, 2, 4],
		backgroundColor: 'rgb(102, 185, 203, 0.6)',
		borderColor: 'rgb(102, 185, 203, 0.9)',
		hoverBackgroundColor: 'rgb(102, 185, 203, 1)',
		verseText: 'I řekl jim: „Dávejte tedy, co je císařovo, císaři, a co jest Božího, Bohu.“'
	},{
		verseId: 'Matt 20:16',
		citations: [5, 4, 3, 2, 1, 2, 4, 4, 6, 6, 3, 2, 2, 0, 2],
		backgroundColor: 'rgb(77, 194, 156, 0.6)',
		borderColor: 'rgb(77, 194, 156, 0.9)',
		hoverBackgroundColor: 'rgb(77, 194, 156, 1)',
		verseText: 'Tak budou poslední prvními a první posledními; neboť mnoho jest povolaných, ale málo vyvolených.“'
	}
];

const dataForPlottingTopEvangelia = 
[{
		verseId: 'Matt 5:38',
		citations: [2, 7, 5, 5, 8, 7, 7, 7, 13, 2, 5, 1, 4, 5, 2],
		backgroundColor: 'rgb(0, 83, 101, 0.6)',
		borderColor: 'rgb(0, 83, 101, 0.9)',
		hoverBackgroundColor: 'rgb(0, 83, 101, 1)',
		verseText: 'Slyšeli jste, že bylo řečeno: „Oko za oko, zub za zub.“'
	},{
		verseId: 'Matt 3:3/John 1:23/Mark 1:3/Luke 3:4/Isa 40:3',
		citations: [4, 9, 11, 6, 5, 6, 9, 1, 4, 0, 3, 1, 1, 0, 5],
		backgroundColor: 'rgb(0, 118, 79, 0.6)',
		borderColor: 'rgb(0, 118, 79, 0.9)',
		hoverBackgroundColor: 'rgb(0, 118, 79, 1)',
		verseText: 'jakož psáno jest v knize řečí proroka Isaiáše: „Hlas volajícího na poušti: Připravte cestu Páně, přímé čiňte stezky jeho;'
	},{
		verseId: 'Luke 2:14',
		citations: [7, 10, 11, 8, 4, 12, 8, 9, 11, 4, 2, 1, 5, 2, 3],
		backgroundColor: 'rgb(159, 101, 30, 0.6)',
		borderColor: 'rgb(159, 101, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 101, 30, 1)',
		verseText: 'Sláva na výsostech Bohu, a na zemi pokoj, lidem dobrá vůle.'
	},{
		verseId: 'Matt 11:28',
		citations: [1, 6, 4, 11, 2, 1, 4, 1, 2, 2, 1, 0, 2, 1, 2],
		backgroundColor: 'rgb(159, 36, 30, 0.6)',
		borderColor: 'rgb(159, 36, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 36, 30, 1)',
		verseText: '„Pojďte ke mně všichni, kteří těžce pracujete a jste přetíženi, a já vám dám odpočinek.'
	},{
		verseId: 'Matt 6:11/Luke 11:3',
		citations: [4, 3, 3, 0, 4, 6, 2, 3, 7, 3, 3, 1, 6, 5, 4],
		backgroundColor: 'rgb(0, 139, 168, 0.6)',
		borderColor: 'rgb(0, 139, 168, 0.9)',
		hoverBackgroundColor: 'rgb(0, 139, 168, 1)',
		verseText: 'Chléb náš vezdejší dej nám dnes.'
	},{
		verseId: 'Matt 27:25',
		citations: [4, 4, 5, 8, 2, 0, 7, 2, 1, 1, 0, 0, 1, 0, 0],
		backgroundColor: 'rgb(0, 168, 113, 0.6)',
		borderColor: 'rgb(0, 168, 113, 0.9)',
		hoverBackgroundColor: 'rgb(0, 168, 113, 1)',
		verseText: 'A odpověděv všecken lid, řekl: Krev jeho na nás, i na naše syny.'
	},{
		verseId: 'Matt 16:18',
		citations: [16, 7, 4, 2, 3, 3, 6, 1, 2, 0, 1, 2, 2, 0, 0],
		backgroundColor: 'rgb(227, 144, 43, 0.6)',
		borderColor: 'rgb(227, 144, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 144, 43, 1)',
		verseText: 'I já pravím tobě: Ty jsi Petr (t. j. skála), a na té skále vzdělám cirkev svou, a brány pekelné jí nepřemohou.'
	},{
		verseId: 'Luke 18:42',
		citations: [8, 1, 6, 2, 5, 1, 1, 4, 2, 1, 0, 0, 1, 1, 2],
		backgroundColor: 'rgb(227, 52, 43, 0.6)',
		borderColor: 'rgb(227, 52, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 52, 43, 1)',
		verseText: 'A Ježíš řekl jemu: Prohlédni. Víra tvá tě uzdravila.'
	},{
		verseId: 'Luke 20:25',
		citations: [6, 4, 3, 5, 1, 4, 5, 0, 1, 2, 2, 1, 2, 2, 4],
		backgroundColor: 'rgb(102, 185, 203, 0.6)',
		borderColor: 'rgb(102, 185, 203, 0.9)',
		hoverBackgroundColor: 'rgb(102, 185, 203, 1)',
		verseText: 'I řekl jim: „Dávejte tedy, co je císařovo, císaři, a co jest Božího, Bohu.“'
	},{
		verseId: 'Matt 20:16',
		citations: [5, 4, 3, 2, 1, 2, 4, 4, 6, 6, 3, 2, 2, 0, 2],
		backgroundColor: 'rgb(77, 194, 156, 0.6)',
		borderColor: 'rgb(77, 194, 156, 0.9)',
		hoverBackgroundColor: 'rgb(77, 194, 156, 1)',
		verseText: 'Tak budou poslední prvními a první posledními; neboť mnoho jest povolaných, ale málo vyvolených.“'
	}
];

const dataForPlottingTopNZ = 
[{
		verseId: '1John 5:6',
		citations: [4, 2, 3, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
		backgroundColor: 'rgb(0, 83, 101, 0.6)',
		borderColor: 'rgb(0, 83, 101, 0.9)',
		hoverBackgroundColor: 'rgb(0, 83, 101, 1)',
		verseText: 'Ježíš Kristus jest ten, který přišel skrze vodu a krev. Nejen s vodou, ale s vodou a s krví. A Duch je toho svědkem, poněvadž Duch je pravda. '
	},{
		verseId: 'Rom 13:11',
		citations: [1, 3, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
		backgroundColor: 'rgb(0, 118, 79, 0.6)',
		borderColor: 'rgb(0, 118, 79, 0.9)',
		hoverBackgroundColor: 'rgb(0, 118, 79, 1)',
		verseText: 'A vědouce ten čas: že již hodina jest, abychom ze sna povstali. Neboť nyní bližší jest naše spasení, nežli když sme uvěřili.'
	},{
		verseId: 'Acts 5:29',
		citations: [4, 2, 2, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
		backgroundColor: 'rgb(159, 101, 30, 0.6)',
		borderColor: 'rgb(159, 101, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 101, 30, 1)',
		verseText: 'Petr a apoštolově odpověděli: „Více sluší poslouchati Boha než lidí.'
	},{
		verseId: '2Pet 1:17',
		citations: [1, 2, 6, 0, 1, 1, 3, 0, 2, 0, 0, 0, 0, 1, 0],
		backgroundColor: 'rgb(159, 36, 30, 0.6)',
		borderColor: 'rgb(159, 36, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 36, 30, 1)',
		verseText: 'Obdržel zajisté od Boha Otce čest i slávu, když se stal k němu takovýto hlas od velikolepé slávy: „Tento jest Syn můj milý, v němž jsem si zalíbil. (Jeho poslouchejte.“)'
	},{
		verseId: 'Exod 20:12/Deut 5:16/Eph 6:2',
		citations: [3, 6, 2, 3, 4, 7, 0, 2, 2, 0, 2, 1, 0, 0, 4],
		backgroundColor: 'rgb(0, 139, 168, 0.6)',
		borderColor: 'rgb(0, 139, 168, 0.9)',
		hoverBackgroundColor: 'rgb(0, 139, 168, 1)',
		verseText: 'Cti otce svého i matku svou, jakož přikázal tobě Hospodin Bůh tvůj, aby se prodleli dnové tvoji, a aby tobě dobře bylo na zemi, kterouž Hospodin Bůh tvůj dá tobě.'
	},{
		verseId: '2Cor 1:2/Phil 1:2/2Thess 1:2/1Cor 1:3/Eph 1:2/Gal 1:3',
		citations: [2, 3, 1, 1, 3, 2, 1, 1, 1, 0, 4, 5, 0, 1, 0],
		backgroundColor: 'rgb(0, 168, 113, 0.6)',
		borderColor: 'rgb(0, 168, 113, 0.9)',
		hoverBackgroundColor: 'rgb(0, 168, 113, 1)',
		verseText: 'Milost vám a pokoj od Boha, našeho Otce, a Pána Ježíše Krista, '
	},{
		verseId: '1Cor 15:20',
		citations: [3, 1, 0, 3, 0, 1, 1, 1, 1, 2, 0, 1, 0, 0, 0],
		backgroundColor: 'rgb(227, 144, 43, 0.6)',
		borderColor: 'rgb(227, 144, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 144, 43, 1)',
		verseText: 'Ale Kristus vstal z mrtvých, jako první ze zesnulých. '
	},{
		verseId: '2Tim 4:7',
		citations: [1, 4, 3, 0, 2, 0, 1, 0, 1, 0, 1, 1, 3, 0, 0],
		backgroundColor: 'rgb(227, 52, 43, 0.6)',
		borderColor: 'rgb(227, 52, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 52, 43, 1)',
		verseText: 'Dobrý boj sem bojoval, běh sem dokonal, víru sem zachoval.'
	},{
		verseId: 'Heb 5:9',
		citations: [2, 2, 1, 1, 1, 1, 1, 3, 3, 1, 0, 2, 0, 0, 0],
		backgroundColor: 'rgb(102, 185, 203, 0.6)',
		borderColor: 'rgb(102, 185, 203, 0.9)',
		hoverBackgroundColor: 'rgb(102, 185, 203, 1)',
		verseText: 'a byv dokonán stal se příčinou věčné spásy všem, kteří ho poslouchají,'
	},{
		verseId: 'Acts 2:17',
		citations: [1, 2, 0, 1, 0, 1, 4, 0, 2, 0, 1, 3, 0, 0, 0],
		backgroundColor: 'rgb(77, 194, 156, 0.6)',
		borderColor: 'rgb(77, 194, 156, 0.9)',
		hoverBackgroundColor: 'rgb(77, 194, 156, 1)',
		verseText: '»V posledních dnech, praví Bůh, vyleji část svého Ducha na všechny lidi; prorokovati budou vaši synové a vaše dcery; vaši jinoši budou míti vidění, vaši starci budou sníti sny;'
	},{
		verseId: '2Cor 2:6',
		citations: [3, 0, 0, 1, 0, 3, 2, 0, 2, 2, 3, 1, 2, 0, 1],
		backgroundColor: 'rgb(235, 177, 107, 0.6)',
		borderColor: 'rgb(235, 177, 107, 0.9)',
		hoverBackgroundColor: 'rgb(235, 177, 107, 1)',
		verseText: 'Toto pokárání od většiny, kterého se mu dostalo, stačí; '
	},{
		verseId: '2Cor 7:10',
		citations: [0, 2, 0, 1, 2, 2, 0, 1, 3, 0, 0, 0, 0, 2, 0],
		backgroundColor: 'rgb(235, 113, 107, 0.6)',
		borderColor: 'rgb(235, 113, 107, 0.9)',
		hoverBackgroundColor: 'rgb(235, 113, 107, 1)',
		verseText: 'Zármutek totiž, který jest podle Boha, působí pokání ke spáse, kterého se nelituje, ale zármutek světa působí smrt.'
	}
];

const dataForPlottingTopSZ = 
[{
		verseId: 'Matt 3:3/John 1:23/Mark 1:3/Luke 3:4/Isa 40:3',
		citations: [4, 9, 11, 6, 5, 6, 9, 1, 4, 0, 3, 1, 1, 0, 5],
		backgroundColor: 'rgb(0, 83, 101, 0.6)',
		borderColor: 'rgb(0, 83, 101, 0.9)',
		hoverBackgroundColor: 'rgb(0, 83, 101, 1)',
		verseText: 'jakož psáno jest v knize řečí proroka Isaiáše: „Hlas volajícího na poušti: Připravte cestu Páně, přímé čiňte stezky jeho;'
	},{
		verseId: 'Exod 20:15/Deut 5:19',
		citations: [6, 3, 7, 3, 5, 3, 6, 1, 1, 2, 2, 2, 2, 0, 4],
		backgroundColor: 'rgb(0, 118, 79, 0.6)',
		borderColor: 'rgb(0, 118, 79, 0.9)',
		hoverBackgroundColor: 'rgb(0, 118, 79, 1)',
		verseText: 'Nepokradeš.'
	},{
		verseId: 'Ps 127:1',
		citations: [2, 2, 0, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
		backgroundColor: 'rgb(159, 101, 30, 0.6)',
		borderColor: 'rgb(159, 101, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 101, 30, 1)',
		verseText: 'Píseň stupňů, Šalomounova. Nebude-li Hospodin stavěti domu, nadarmo usilují ti, kteříž stavějí jej; nebude-li Hospodin ostříhati města, nadarmo bdí strážný.'
	},{
		verseId: 'Exod 20:12/Deut 5:16/Eph 6:2',
		citations: [3, 6, 2, 3, 4, 7, 0, 2, 2, 0, 2, 1, 0, 0, 4],
		backgroundColor: 'rgb(159, 36, 30, 0.6)',
		borderColor: 'rgb(159, 36, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 36, 30, 1)',
		verseText: 'Cti otce svého i matku svou, jakož přikázal tobě Hospodin Bůh tvůj, aby se prodleli dnové tvoji, a aby tobě dobře bylo na zemi, kterouž Hospodin Bůh tvůj dá tobě.'
	},{
		verseId: 'Gen 3:19',
		citations: [4, 1, 1, 0, 5, 3, 1, 0, 2, 2, 0, 0, 0, 0, 0],
		backgroundColor: 'rgb(0, 139, 168, 0.6)',
		borderColor: 'rgb(0, 139, 168, 0.9)',
		hoverBackgroundColor: 'rgb(0, 139, 168, 1)',
		verseText: 'V potu tváře jísti budeš chléb, dokud se nevrátíš do země, ze které jsi vzat. Ano, prach jsi, a v prach se navrátíš.'
	},{
		verseId: 'Prov 26:27',
		citations: [3, 4, 2, 2, 3, 4, 3, 1, 2, 1, 0, 2, 3, 5, 1],
		backgroundColor: 'rgb(0, 168, 113, 0.6)',
		borderColor: 'rgb(0, 168, 113, 0.9)',
		hoverBackgroundColor: 'rgb(0, 168, 113, 1)',
		verseText: 'Kdo (jinému) jámu kopá, (sám) do ní padá, a kdo valí kámen, bude ním zavalen.'
	},{
		verseId: 'Exod 20:13/Deut 5:17',
		citations: [11, 6, 11, 6, 16, 7, 6, 7, 5, 11, 6, 7, 2, 3, 3],
		backgroundColor: 'rgb(227, 144, 43, 0.6)',
		borderColor: 'rgb(227, 144, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 144, 43, 1)',
		verseText: 'Nezabiješ.'
	},{
		verseId: '2Sam 14:5',
		citations: [0, 1, 2, 1, 0, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1],
		backgroundColor: 'rgb(227, 52, 43, 0.6)',
		borderColor: 'rgb(227, 52, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 52, 43, 1)',
		verseText: 'I tázal se jí král: „Co je ti?" A ona odpověděla: „Ach, vdovou jsem já, neboť umřel mi muž.'
	},{
		verseId: 'Exod 20:14/Deut 5:18',
		citations: [3, 0, 3, 3, 3, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
		backgroundColor: 'rgb(102, 185, 203, 0.6)',
		borderColor: 'rgb(102, 185, 203, 0.9)',
		hoverBackgroundColor: 'rgb(102, 185, 203, 1)',
		verseText: 'Nesesmilníš.'
	},{
		verseId: 'Isa 45:8',
		citations: [1, 0, 0, 1, 4, 2, 0, 0, 0, 3, 1, 0, 1, 0, 0],
		backgroundColor: 'rgb(77, 194, 156, 0.6)',
		borderColor: 'rgb(77, 194, 156, 0.9)',
		hoverBackgroundColor: 'rgb(77, 194, 156, 1)',
		verseText: 'Rosu dejte nebesa s hůry, a nejvyšší oblakové dštěte spravedlnost; otevři se země, a ať vzejde spasení, a spravedlnost ať spolu vykvete. Já Hospodin způsobím to.'
	},{
		verseId: 'Prov 31:10',
		citations: [1, 2, 2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
		backgroundColor: 'rgb(235, 177, 107, 0.6)',
		borderColor: 'rgb(235, 177, 107, 0.9)',
		hoverBackgroundColor: 'rgb(235, 177, 107, 1)',
		verseText: 'Ženu statečnou kdo nalezne? Nebo daleko nad perly cena její.'
	},{
		verseId: 'Ezek 24:19',
		citations: [1, 3, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1, 1, 1, 0],
		backgroundColor: 'rgb(235, 113, 107, 0.6)',
		borderColor: 'rgb(235, 113, 107, 0.9)',
		hoverBackgroundColor: 'rgb(235, 113, 107, 1)',
		verseText: 'I řekl mně lid: „Proč nám neoznamuješ, co znamenají tyto věci, jež činíš?"'
	}
];

const dataForPlottingTopDesatero = 
[{
		verseId: 'Exod 20:2/Deut 5:6',
		citations: [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		backgroundColor: 'rgb(0, 83, 101, 0.6)',
		borderColor: 'rgb(0, 83, 101, 0.9)',
		hoverBackgroundColor: 'rgb(0, 83, 101, 1)',
		verseText: 'Já jsem Hospodin Bůh tvůj, kterýž jsem tě vyvedl z země Egyptské z domu služby.'
	},{
		verseId: 'Exod 20:5',
		citations: [0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		backgroundColor: 'rgb(0, 118, 79, 0.6)',
		borderColor: 'rgb(0, 118, 79, 0.9)',
		hoverBackgroundColor: 'rgb(0, 118, 79, 1)',
		verseText: 'Nebudeš se jim klaněti, aniž (jich) ctíti; jáť jsem Hospodin, Bůh tvůj silný a řevnivý, jenž do třetího a čtvrtého kolena stíhá syny za nepravosti těch otců, kteří mne nenávidí,'
	},{
		verseId: 'Exod 20:15/Deut 5:19',
		citations: [6, 3, 7, 3, 5, 3, 6, 1, 1, 2, 2, 2, 2, 0, 4],
		backgroundColor: 'rgb(159, 101, 30, 0.6)',
		borderColor: 'rgb(159, 101, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 101, 30, 1)',
		verseText: 'Nepokradeš.'
	},{
		verseId: 'Exod 20:12/Deut 5:16/Eph 6:2',
		citations: [3, 6, 2, 3, 4, 7, 0, 2, 2, 0, 2, 1, 0, 0, 4],
		backgroundColor: 'rgb(159, 36, 30, 0.6)',
		borderColor: 'rgb(159, 36, 30, 0.9)',
		hoverBackgroundColor: 'rgb(159, 36, 30, 1)',
		verseText: 'Cti otce svého i matku svou, jakož přikázal tobě Hospodin Bůh tvůj, aby se prodleli dnové tvoji, a aby tobě dobře bylo na zemi, kterouž Hospodin Bůh tvůj dá tobě.'
	},{
		verseId: 'Exod 20:13/Deut 5:17',
		citations: [11, 6, 11, 6, 16, 7, 6, 7, 5, 11, 6, 7, 2, 3, 3],
		backgroundColor: 'rgb(0, 139, 168, 0.6)',
		borderColor: 'rgb(0, 139, 168, 0.9)',
		hoverBackgroundColor: 'rgb(0, 139, 168, 1)',
		verseText: 'Nezabiješ.'
	},{
		verseId: 'Exod 20:3/Deut 5:7',
		citations: [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
		backgroundColor: 'rgb(0, 168, 113, 0.6)',
		borderColor: 'rgb(0, 168, 113, 0.9)',
		hoverBackgroundColor: 'rgb(0, 168, 113, 1)',
		verseText: 'Nebudeš míti bohů jiných přede mnou.'
	},{
		verseId: 'Exod 20:14/Deut 5:18',
		citations: [3, 0, 3, 3, 3, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
		backgroundColor: 'rgb(227, 144, 43, 0.6)',
		borderColor: 'rgb(227, 144, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 144, 43, 1)',
		verseText: 'Nesesmilníš.'
	},{
		verseId: 'Deut 5:21',
		citations: [3, 2, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
		backgroundColor: 'rgb(227, 52, 43, 0.6)',
		borderColor: 'rgb(227, 52, 43, 0.9)',
		hoverBackgroundColor: 'rgb(227, 52, 43, 1)',
		verseText: 'Nepožádáš manželky bližního svého, aniž požádáš domu bližního svého, pole jeho, neb služebníka jeho, aneb děvky jeho, vola jeho neb osla jeho, aneb čehokoli z těch věcí, kteréž jsou bližního tvého.'
	},{
		verseId: 'Exod 20:16/Deut 5:20',
		citations: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		backgroundColor: 'rgb(102, 185, 203, 0.6)',
		borderColor: 'rgb(102, 185, 203, 0.9)',
		hoverBackgroundColor: 'rgb(102, 185, 203, 1)',
		verseText: 'Nepromluvíš proti bližnímu svému křivého svědectví.'
	},{
		verseId: 'Deut 5:9',
		citations: [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		backgroundColor: 'rgb(77, 194, 156, 0.6)',
		borderColor: 'rgb(77, 194, 156, 0.9)',
		hoverBackgroundColor: 'rgb(77, 194, 156, 1)',
		verseText: 'Nebudeš se jim klaněti, ani jich ctíti. Nebo já jsem Hospodin Bůh tvůj, Bůh silný, horlivý, navštěvující nepravost otců na synech do třetího i čtvrtého pokolení těch, kteříž nenávidí mne,'
	},{
		verseId: 'Exod 20:8',
		citations: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		backgroundColor: 'rgb(235, 177, 107, 0.6)',
		borderColor: 'rgb(235, 177, 107, 0.9)',
		hoverBackgroundColor: 'rgb(235, 177, 107, 1)',
		verseText: 'Pomni na den sobotní, abys jej světil.'
	}
];

function makeDatasets(value, index, array) {
    let datasetInstance = {
        label: array[index]['verseId'],
        backgroundColor: array[index]['backgroundColor'],
        borderColor: array[index]['borderColor'],
        hoverBackgroundColor: array[index]['hoverBackgroundColor'],
        borderWidth: borderWidth,
        data: array[index]['citations'],
        verseText: array[index]['verseText'],
        //data: {x: labels, y: array[index]['citations'], status: array[index]['verseText']},
        tension: tension
    }
    return datasetInstance;
}

let datasets = dataForPlottingTop.map(makeDatasets);

let data = {
    labels: labels,
    datasets: datasets
  };

const barOptions = {
    scales: {
        y: {
            beginAtZero: true
        },
        x: {
            min: 0,
            max: 15
        }
    },
    responsive: true,
	aspectRatio: 2.1,
	maintainAspectRatio: false,
    skipNull: false,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
        },
        tooltip: {
            callbacks: {
                label: (context) => {
					var multistringText = [];
					multistringText.push(`${context.dataset.label}`)
					multistringText.push(`Text: ${context.dataset.verseText}`)
					multistringText.push(`Počet citací: ${context.parsed.y}`)
					return multistringText
                }
            }
        }
    }
};

const lineOptions = {
    scales: {
        y: {
            beginAtZero: true
        },
        x: {
            min: 0,
            max: 15
        }
    },
    responsive: true,
	aspectRatio: 2.1,
	maintainAspectRatio: false,
    skipNull: false,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
        },
        tooltip: {
            callbacks: {
                label: (context) => {
					var multistringText = [];
					multistringText.push(`${context.dataset.label}`)
					multistringText.push(`Text: ${context.dataset.verseText}`)
					multistringText.push(`Počet citací: ${context.parsed.y}`)
					return multistringText
                }
            }
        }
    },
    fill: false,
    interaction: {
        intersect: false
    },
    radius: 0,
}

const barHorizontalOptions = {
    aspectRatio: 0.8,
    indexAxis: 'y',
    scales: {
        y: {
            min: 0,
            max: 15
        },
        x: {
            beginAtZero: true
        }
    },
    responsive: true,
    skipNull: true,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
        },
        tooltip: {
            callbacks: {
                label: (context) => {
					var multistringText = [];
					multistringText.push(`${context.dataset.label}`)
					multistringText.push(`Text: ${context.dataset.verseText}`)
					multistringText.push(`Počet citací: ${context.parsed.y}`)
					return multistringText
                }
            }
        }
    }
}

const lineHorizontalOptions = {
    aspectRatio: 0.8,
    indexAxis: 'y',
    scales: {
        y: {
            min: 0,
            max: 15
        },
        x: {
            beginAtZero: true
        }
    },
    responsive: true,
    skipNull: true,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
        },
        tooltip: {
            callbacks: {
                label: (context) => {
					var multistringText = [];
					multistringText.push(`${context.dataset.label}`)
					multistringText.push(`Text: ${context.dataset.verseText}`)
					multistringText.push(`Počet citací: ${context.parsed.y}`)
					return multistringText
                }
            }
        }
    },
    fill: false,
    interaction: {
        intersect: false
    },
    radius: 0,
}

// Top citations in years chart:
let configTopInYears = {
    type: 'line',
    data: data,
    options: lineOptions,
};

let chartCitationsTopInYears = new Chart(
    document.getElementById('chartCitationsTopInYears'),
    configTopInYears
);

// Change year scale
function changeYearsTopInYears(start, end, chartOrientation) {
    if (chartOrientation == 'vertical') {
        chartCitationsTopInYears.options.scales.x.min = start;
        chartCitationsTopInYears.options.scales.x.max = end;
        chartCitationsTopInYears.update();
    } else if (chartOrientation == 'horizontal') {
        chartCitationsTopInYears.options.scales.y.min = start;
        chartCitationsTopInYears.options.scales.y.max = end;
        chartCitationsTopInYears.update();
    };
}

// Apply slider change
let yearSlider = document.getElementById('slider-round');

noUiSlider.create(yearSlider, {
    start: [1925, 1939],
    connect: true,
    range: {
        'min': 1925,
        'max': 1939
    },
    connect: true,
    format: wNumb({
        decimals: 0
    }),
    step: 1,
    tooltips: true
});

let minYearId = 0;
let maxYearId = 15+1;

yearSlider.noUiSlider.on('update', function (values, handle) {
    if (handle) {
        let maxYear = parseInt(values[handle]);
        maxYearId = maxYear-1925;
    } else {
        let minYear = parseInt(values[handle]);
        minYearId = minYear-1925;
    }
});

let currentChartOrientation = 'vertical';
yearSlider.noUiSlider.on('update', function() {changeYearsTopInYears(minYearId, maxYearId, currentChartOrientation)});

// Change chart type:
let barLineButton = document.getElementById('BarLineButton');

function changeChartType() {
    chartCitationsTopInYears.destroy();
    if (configTopInYears.options == barOptions) {
        configTopInYears.type = 'line';
        configTopInYears.options = lineOptions;
        barLineButton.innerHTML = 'sloupcový graf';
    } else if (configTopInYears.options == lineOptions) {
        configTopInYears.type = 'bar';
        barLineButton.innerHTML = 'čárový graf';
        configTopInYears.options = barOptions;
    } else if (configTopInYears.options == lineHorizontalOptions) {
        configTopInYears.type = 'bar';
        barLineButton.innerHTML = 'čárový graf';
        configTopInYears.options = barHorizontalOptions;
    } else if (configTopInYears.options == barHorizontalOptions) {
        configTopInYears.type = 'line';
        barLineButton.innerHTML = 'sloupcový graf';
        configTopInYears.options = lineHorizontalOptions;
    };
    chartCitationsTopInYears = new Chart(
        document.getElementById('chartCitationsTopInYears'),
        configTopInYears);
    changeYearsTopInYears(minYearId,maxYearId);
    chartCitationsTopInYears.update();
}

barLineButton.onclick = function() {changeChartType()};

// Change selected dataset
let selectedDataset = document.getElementById('selectDataset');

let commentTop10 = '<h3>The most frequent citations in total</h3><p>The chart shows how the citation rate of the ten most referenced parts of the Bible in the corpus evolved. There is a persistent interest in the Ten Commandments, where the commandment „<i>Nezabiješ</i>“ (Ex 20:13/Dt 5:17), is the most quoted verse overall, and in the gospels, where the most quoted passage overall is Luke 2:14: „<i>Sláva na výsostech bohu</i>“. You can find an overview of the most frequent citations in total <a href="https://public.flourish.studio/visualisation/11097410/">zde</a></p>';

let commentTopGospels = '<h3>The most frequent citations from the gospels</h3><p>The chart shows how the citation rate of the ten most referenced parts of the Bible evolved, this time focusing only on the gospels.</p>';

let commentTopNT = '<h3>The most frequent citations from the New Testament</h3><p>Again, the chart shows how the citation rate of the ten most referenced parts of the Bible evolved, this time focusing only on the New Testament, except the gospels. Here, the statistics also include the commandment „<i>Cti otce svého i matku svou</i>“, na které v Novém zákoně odkazuje také verš Ef 6:2.</p>';

let commentTopOT = '<h3>The most frequent citations from the Old Testament</h3><p>The chart shows how the citation rate of the ten most referenced citations from the Old Testament evolved. Also interesting is the joint occurrence of the moralising citations (Gen 3:19, Prov. 26:27, Iza 45:8), which have mostly become parts of folklore.</p>';

let commentTopTC = '<h3>The most frequent citations from the Ten Commandments</h3><p>The chart shows which of the Ten Commandments were quoted most frequently in the corpus, as well as changes in their citation rate over time. „<i>Nezabiješ</i>“ (Ex 20:13 / Dt 5:17) is the most frequently quoted commandment. The overall appearance of the commandment „<i>Nepokradeš</i>“ (Ex 20:15 / Dt 5:19) is less frequent, but nevertheless still persistent, illustrating the general social and political situation.</p>';

let commentSection = document.getElementById('commentSection');

function changeDataset() {
    if (selectedDataset.value == 'Top') {
        datasets = dataForPlottingTop.map(makeDatasets);
        chartCitationsTopInYears.data.datasets = datasets;
        chartCitationsTopInYears.update();
        commentSection.innerHTML = commentTop10
        console.log(datasets);
    } else if (selectedDataset.value == 'TopEvangelia') {
        datasets = dataForPlottingTopEvangelia.map(makeDatasets);
        chartCitationsTopInYears.data.datasets = datasets;
        chartCitationsTopInYears.update();
        commentSection.innerHTML = commentTopGospels
        console.log(configTopInYears.datasets);
    } else if (selectedDataset.value == 'TopNZ') {
        datasets = dataForPlottingTopNZ.map(makeDatasets);
        chartCitationsTopInYears.data.datasets = datasets;
        chartCitationsTopInYears.update();
        commentSection.innerHTML = commentTopNT
        console.log(configTopInYears.datasets);
    } else if (selectedDataset.value == 'TopSZ') {
        datasets = dataForPlottingTopSZ.map(makeDatasets);
        chartCitationsTopInYears.data.datasets = datasets;
        chartCitationsTopInYears.update();
        commentSection.innerHTML = commentTopOT
        console.log(configTopInYears.datasets);
    } else if (selectedDataset.value == 'TopDesatero') {
        datasets = dataForPlottingTopDesatero.map(makeDatasets);
        chartCitationsTopInYears.data.datasets = datasets;
        chartCitationsTopInYears.update();
        commentSection.innerHTML = commentTopTC
        console.log(configTopInYears.datasets);
    };
    
}

selectedDataset.onchange = function() {changeDataset()};

// Rotate chart:
let rotateChartButton = document.getElementById('RotateChartButton');

function RotateChart() {
    chartCitationsTopInYears.destroy();
    if (configTopInYears.options == barOptions) {
        configTopInYears.type = 'bar';
        configTopInYears.options = barHorizontalOptions;
        currentChartOrientation = 'horizontal';
    } else if (configTopInYears.options == lineOptions) {
        configTopInYears.type = 'line';
        configTopInYears.options = lineHorizontalOptions;
        currentChartOrientation = 'horizontal';
    } else if (configTopInYears.options == lineHorizontalOptions) {
        configTopInYears.type = 'line';
        configTopInYears.options = lineOptions;
        currentChartOrientation = 'vertical';
    } else if (configTopInYears.options == barHorizontalOptions) {
        configTopInYears.type = 'bar';
        configTopInYears.options = barOptions;
        currentChartOrientation = 'vertical';
    };
    chartCitationsTopInYears = new Chart(
        document.getElementById('chartCitationsTopInYears'),
        configTopInYears);
    changeYearsTopInYears(minYearId,maxYearId);
    chartCitationsTopInYears.update();
}

rotateChartButton.onclick = function() {RotateChart()};

function unrotateWithWidth() {
    if (window.matchMedia("(max-width: 550px)").matches) {
        console.log('mobile charts')
    } else {
        if (configTopInYears.options == lineHorizontalOptions) {
            chartCitationsTopInYears.destroy();

            configTopInYears.type = 'line';
            configTopInYears.options = lineOptions;
            currentChartOrientation = 'vertical';
            
            chartCitationsTopInYears = new Chart(
                document.getElementById('chartCitationsTopInYears'),
                configTopInYears);
            changeYearsTopInYears(minYearId,maxYearId);
            chartCitationsTopInYears.update();
        } else if (configTopInYears.options == barHorizontalOptions) {
            chartCitationsTopInYears.destroy();
            
            configTopInYears.type = 'bar';
            configTopInYears.options = barOptions;
            currentChartOrientation = 'vertical';
            
            chartCitationsTopInYears = new Chart(
                document.getElementById('chartCitationsTopInYears'),
                configTopInYears);
            changeYearsTopInYears(minYearId,maxYearId);
            chartCitationsTopInYears.update();
        };
    };
}

window.addEventListener("resize", function() {unrotateWithWidth()});

// Below it is prepared for individual graphs (not used in the end)
// // Top10 - Config
// let datasetTop10 = dataForPlottingTop.map(makeDatasets);

// let dataTop10 = {
//     labels: labels,
//     datasets: datasetTop10
//   };

// let configTop10 = {
//     type: 'line',
//     data: dataTop10,
//     options: lineOptions,
// };

// let chartCitationsTop = new Chart(
//     document.getElementById('chartCitationsTop'),
//     configTop10
// );

// // TopGospels - Config
// let datasetTopGospels = dataForPlottingTopEvangelia.map(makeDatasets);

// let dataTopGospels = {
//     labels: labels,
//     datasets: datasetTopGospels
//   };

// let configTopGospels = {
//     type: 'line',
//     data: dataTopGospels,
//     options: lineOptions,
// };

// let chartCitationsTopGospels = new Chart(
//     document.getElementById('chartCitationsTopGospels'),
//     configTopGospels
// );

// // TopNT - Config
// let datasetTopNT = dataForPlottingTopNZ.map(makeDatasets);

// let dataTopNT = {
//     labels: labels,
//     datasets: datasetTopNT
//   };

// let configTopNT = {
//     type: 'line',
//     data: dataTopNT,
//     options: lineOptions,
// };

// let chartCitationsTopNT = new Chart(
//     document.getElementById('chartCitationsTopNT'),
//     configTopNT
// );

// // TopOT - Config
// let datasetTopOT = dataForPlottingTopSZ.map(makeDatasets);

// let dataTopOT = {
//     labels: labels,
//     datasets: datasetTopOT
//   };

// let configTopOT = {
//     type: 'line',
//     data: dataTopOT,
//     options: lineOptions,
// };

// let chartCitationsTopOT = new Chart(
//     document.getElementById('chartCitationsTopOT'),
//     configTopOT
// );

// // TopTC - Config
// let datasetTopTC = dataForPlottingTopDesatero.map(makeDatasets);

// let dataTopTC = {
//     labels: labels,
//     datasets: datasetTopTC
//   };

// let configTopTC = {
//     type: 'line',
//     data: dataTopTC,
//     options: lineOptions,
// };

// let chartCitationsTopTC = new Chart(
//     document.getElementById('chartCitationsTopTC'),
//     configTopTC
// );