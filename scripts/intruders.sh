grep '@liferay-headless' .github/CODEOWNERS | awk '{print $1}' | while read -r folder; do
    commits=$(git log $1..HEAD --pretty=format:'%h %ae %s' -- "$folder" | grep -v -E 'brian.chan@liferay.com|alejandro.tardin@liferay.com|alberto.moreno@liferay.com|jalber786@gmail.com|carlos.correa@liferay.com|daniel.raposo@liferay.com|daniel.szimko@liferay.com|jaime.leon@liferay.com|magdalena.jedraszak@liferay.com|vendel.toreki@liferay.com|gabor.komaromi@liferay.com|petteri.karttunen@liferay.com|jorge.gonzalez@liferay.com|joseluis.navarro@liferay.com')
    if [ -n "$commits" ]; then
        echo "#### Intruders in $folder"
        echo "$commits"
        echo ""
    fi
done
